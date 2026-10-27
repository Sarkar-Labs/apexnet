here"""Trainer: everything about *training* lives here, models stay pure.

Features: multi-step tf.function loops, global-batch-correct loss normalisation,
gradient accumulation, global-norm clipping, NaN/Inf step skipping, bf16/fp16
(with dynamic loss scaling), EMA weights, tf.distribute, checkpoint/resume,
SavedModel export, and a one-method override point (``train_step``).
"""
import contextlib
import functools
import os
import numpy as np
import tensorflow as tf

from . import data as D
from .callbacks import CallbackList, Logger
from .core import Module, _Policy, set_precision

SUM = tf.distribute.ReduceOp.SUM
FIRST = tf.VariableAggregation.ONLY_FIRST_REPLICA


def _var(value, dtype):
    return tf.Variable(value, dtype=dtype, trainable=False, aggregation=FIRST)


class LossScaler(tf.Module):
    """Dynamic loss scaling for float16."""

    def __init__(self, init=2.0 ** 15, period=2000, factor=2.0):
        super().__init__()
        self.period, self.factor = period, factor
        self.scale, self.good = _var(init, tf.float32), _var(0, tf.int64)

    def update(self, finite):
        g = self.good + 1
        grow = finite & (g >= self.period)
        up = tf.minimum(self.scale * self.factor, 2.0 ** 24)
        self.scale.assign(tf.where(finite, tf.where(grow, up, self.scale),
                                   tf.maximum(self.scale / self.factor, 1.0)))
        self.good.assign(tf.where(finite & ~grow, g, tf.zeros_like(g)))


class _Feed:
    """A (distributed) dataset iterator that tracks how many batches remain."""

    def __init__(self, make_iter, card):
        self.make_iter, self.card = make_iter, card
        self.reset()

    def reset(self):
        self.it = self.make_iter()
        self.remaining = self.card


class _Meter:
    """Streaming accumulators (scalars or fixed-shape vectors) living in TF variables.
    ``spec``: name -> tuple of shapes, one per statistic the metric emits."""

    def __init__(self, spec):
        self.spec = spec
        self.v = {n: [_var(tf.zeros(sh, tf.float64), tf.float64) for sh in shapes] for n, shapes in spec.items()}

    def update(self, stats):
        for n, vals in stats.items():
            for var, x in zip(self.v[n], vals):
                var.assign_add(tf.cast(x, tf.float64))

    def reset(self):
        for vs in self.v.values():
            for var in vs:
                var.assign(tf.zeros_like(var))

    def result(self, metrics, loss_name="loss"):
        import numpy as np
        out = {}
        for n, vs in self.v.items():
            sums = [x.numpy() for x in vs]
            if n in metrics:
                out[n] = float(metrics[n].result(*[x if x.ndim else float(x) for x in sums]))
            else:  # loss and extra scalar stats: mean = sum / count
                out[n] = float(sums[0]) / max(float(sums[1]), 1e-12)
        return out


def _own_precision(fn):
    """Run a Trainer method under THAT trainer's precision policy, then restore the previous global policy.
    (The policy is process-global state; without this, constructing another Trainer silently changed the dtype of
    this one's fit / evaluate / predict.)"""
    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        old = _Policy.compute_dtype
        set_precision(self.precision_name)
        try:
            return fn(self, *a, **k)
        finally:
            _Policy.compute_dtype = old
    return wrapper


def _unpack(batch):
    if len(batch) == 2:
        return batch[0], batch[1], None
    if len(batch) == 3:
        return batch
    raise ValueError("batches must be (x, y) or (x, y, sample_weight)")


def _is_dataset(d):
    return isinstance(d, tf.data.Dataset) or hasattr(d, "element_spec")


class Trainer:
    def __init__(self, model, loss, optimizer=None, metrics=None, *, grad_clip=None, accumulate=1,
                 precision="float32", ema=None, strategy=None, jit="auto", steps_per_call=1,
                 skip_nonfinite=True, extra_optimizers=None, extra_stats=(), reg_weight=1.0, sam=None,
                 build_mode=None, static_batches=None):
        """``extra_optimizers``: {name: (optimizer, lambda model: variables)} for multi-optimizer setups
        (GANs, two-tower, LR-per-group); use ``self.apply_gradients`` in a custom ``train_step``.
        ``extra_stats``: names of extra scalar means your custom step reports (see ``_stats``).
        ``sam``: radius rho for Sharpness-Aware Minimization (two forward/backward passes per step).
        Hardware knobs: ``precision``/``jit``/``steps_per_call`` accept "auto" (TPU: bf16; GPU: bf16 on Ampere+, fp16 on
        Volta/Turing; XLA on a single GPU; more steps per call on accelerators). ``static_batches`` (default: on for TPU)
        drops the ragged final training batch and zero-pads eval/predict batches (padding never affects results).
        ``build_mode``: "eager" (default) or "trace" (default on TPU: variables are created by tracing, without
        running the forward pass on the host)."""
        from . import hardware as _hw
        if isinstance(strategy, str):
            if strategy != "auto":
                raise ValueError("strategy must be a tf.distribute.Strategy, a Hardware object, or 'auto'")
            strategy = _hw.setup(verbose=False)
        if isinstance(strategy, _hw.Hardware):
            self.hardware, strategy = strategy, strategy.strategy
        self.strategy = strategy or tf.distribute.get_strategy()
        self.is_tpu = "TPU" in type(self.strategy).__name__
        _gi = _hw.gpu_info()
        self.kind = "tpu" if self.is_tpu else ("gpu" if _gi else "cpu")
        if precision == "auto":
            precision = _hw.recommend_precision(self.kind, _gi[0]["compute_capability"] if _gi else None,
                                                _hw.os_info()["system"])
        if steps_per_call == "auto":
            steps_per_call = _hw.recommend_steps_per_call(self.kind)
        if jit == "auto":  # XLA: single GPU only (TPU always compiles; collectives inside nested XLA are fragile)
            jit = (self.kind == "gpu" and self.strategy.num_replicas_in_sync == 1
                   and _hw.os_info()["system"] != "macos")  # no XLA on the Metal plugin
        jit = bool(jit) and not self.is_tpu
        self.jit_note, self._xla_checked = None, {"train": False, "eval": False}
        self.cond_updates = self.strategy.num_replicas_in_sync == 1 and not self.is_tpu
        needs_static = any(getattr(m, "requires_static_shapes", False) for m in (model, *model.submodules))
        self.static_batches = (self.is_tpu or needs_static) if static_batches is None else static_batches  # e.g. an.Jax layers
        self.build_mode = build_mode or ("trace" if self.is_tpu else "eager")
        self.ckpt_opts = _hw.checkpoint_options(self.strategy)
        set_precision(precision)
        self.precision_name = precision
        self.precision, self.model, self.loss_fn, self.opt = precision, model, loss, optimizer
        self.extra_opts = dict(extra_optimizers or {})
        self.extra_stats, self.reg_weight, self.sam = list(extra_stats), reg_weight, sam
        assert optimizer is not None or self.extra_opts, "need an optimizer or extra_optimizers"
        if metrics is None:
            metrics = {}
        if not isinstance(metrics, dict):
            metrics = {m.name: m for m in metrics}
        self.metrics = metrics
        self.loss_name = "loss"
        self.grad_clip, self.accumulate, self.ema_decay = grad_clip, accumulate, ema
        self.jit, self.steps_per_call, self.skip_nonfinite = jit, steps_per_call, skip_nonfinite
        self.scaler = None  # created inside the strategy scope at init (needs mirrored variables)
        self._want_scaler = precision == "float16"
        self.history, self.stop, self.target_epochs = {}, False, 0
        self._ready, self._pending = False, None

    @classmethod
    def auto(cls, model, loss, optimizer=None, **kw):
        """Everything automatic: detect OS + hardware (TPU > GPU > CPU), choose strategy / precision / XLA /
        steps_per_call, and build the Trainer. The detected ``Hardware`` is kept as ``trainer.hardware``."""
        from . import hardware as _hw
        hw = kw.pop("hardware", None) or _hw.setup(verbose=kw.pop("verbose", False))
        tr = cls(model, loss, optimizer, **{**hw.trainer_kwargs(), **kw})
        tr.hardware = hw
        return tr

    # ------------------------------------------------------------ init/state
    def _ensure_init(self, x):
        if self._ready:
            return
        with self.strategy.scope():
            if self._want_scaler:
                self.scaler = LossScaler()
            self._adopt_variables()
            self._build_model(x)
            tv = self.model.trainable_variables
            for o in self.optimizers:
                o._state()  # counters / lr_scale must be created under the strategy scope
            if self.opt is not None:
                self.opt.build(tv)
            self.opt_vars = {}
            for k, (o, sel) in self.extra_opts.items():
                self.opt_vars[k] = list(sel(self.model))
                o.build(self.opt_vars[k])
            self.acc = [tf.Variable(tf.zeros(v.shape), trainable=False, aggregation=FIRST) for v in tv] \
                if self.accumulate > 1 else []
            self.micro, self.skipped = _var(0, tf.int64), _var(0, tf.int64)  # int64 (not int32): int32 variables are a GPU placement hazard
            self.last_norm, self._epoch = _var(0.0, tf.float32), _var(0, tf.int64)
            self.ema_vars = [tf.Variable(v.read_value(), trainable=False, aggregation=FIRST)
                             for v in self.model.variables] if self.ema_decay else []
            if self.opt is None:  # primary optimizer slot unused in multi-optimizer mode
                self.acc = []
            self.state = tf.Module()
            self.state.model, self.state.opt = self.model, self.opt
            self.state.extra_opts = {k: o for k, (o, _) in self.extra_opts.items()}
            self.state.aux = dict(acc=self.acc, ema=self.ema_vars, micro=self.micro,
                                  skipped=self.skipped, epoch=self._epoch)
            if self.scaler is not None:
                self.state.aux["scaler"] = self.scaler
            self._ckpt = tf.train.Checkpoint(state=self.state)
        # Metric accumulators live on the HOST (outside the strategy scope): float64 is unsupported on TPU and they
        # are only touched in cross-replica code, after strategy.reduce.
        spec = {self.loss_name: ((), ())}
        spec.update({n: tuple(m.shapes) for n, m in self.metrics.items()})
        spec.update({n: ((), ()) for n in self.extra_stats})
        with tf.device("/CPU:0"):
            self.train_meter, self.eval_meter = _Meter(spec), _Meter(spec)
        self._build_loops()
        self._ready = True
        if self._pending:
            self._restore(self._pending)
            self._pending = None

    def _build_loops(self):
        jit = (lambda f: tf.function(f, jit_compile=True)) if self.jit else (lambda f: f)
        self._train_loop = self._make_loop(jit(self.train_step), self.train_meter)
        self._eval_loop = self._make_loop(jit(self.eval_step), self.eval_meter)

    def _verify_xla(self, which, batch):
        """XLA safety net. Compile (never execute) the step for this exact batch; if XLA cannot compile the model
        (tf.py_function / an.Torch, string ops, some Keras layers, ...), fall back to normal execution instead of
        crashing on the first step. Only GPU gets XLA by default, so this is what keeps 'works on CPU' == 'works on GPU'."""
        if not self.jit or self._xla_checked[which]:
            return
        fn = self.train_step if which == "train" else self.eval_step
        try:
            tf.function(fn, jit_compile=True).experimental_get_compiler_ir(batch)(stage="hlo")
            self._xla_checked[which] = True
        except Exception as e:  # noqa
            self.jit = False
            self.jit_note = f"XLA disabled: {type(e).__name__}: {' '.join(str(e).split())[:200]}"
            print("apexnet: " + self.jit_note + " -- running this model without XLA")
            self._build_loops()

    def settings(self):
        """What this Trainer resolved from 'auto' (paste this when reporting a hardware problem)."""
        d = {"device": self.kind, "strategy": type(self.strategy).__name__, "replicas": self.strategy.num_replicas_in_sync,
             "precision": self.precision_name, "xla_jit": self.jit, "jit_note": self.jit_note,
             "steps_per_call": self.steps_per_call, "static_batches": self.static_batches,
             "build_mode": self.build_mode, "cond_updates": self.cond_updates,
             "gpus": [g.name for g in tf.config.list_physical_devices("GPU")], "tensorflow": tf.__version__}
        print(d)
        return d

    def _adopt_variables(self):
        """Variables made in a Module's __init__ (``self.pos = self.param(...)``) exist before the strategy scope is
        entered. Under MirroredStrategy/TPU they must be distributed variables, so re-create them here (same value,
        name, trainable flag) and repoint the owning module + its regularizer list."""
        if self.strategy.num_replicas_in_sync == 1 and not self.is_tpu:
            return 0
        ext, n = self.strategy.extended, 0
        for m in (self.model, *self.model.submodules):
            for k, v in list(vars(m).items()):
                if isinstance(v, tf.Variable) and not ext.variable_created_in_scope(v):
                    new = tf.Variable(v.read_value(), trainable=v.trainable, name=v.name.split(":")[0],
                                      dtype=v.dtype, aggregation=v.aggregation)
                    setattr(m, k, new)
                    if getattr(m, "_regs", None):
                        m._regs = [(new if rv is v else rv, fn) for rv, fn in m._regs]
                    n += 1
        return n

    def _build_model(self, x):
        """Create every variable. 'trace' builds by tracing a tf.function (no host forward compute: needed on TPU);
        if tracing fails the half-built modules are reset and the model is built eagerly instead."""
        self.build_info = self.build_mode
        if self.build_mode == "trace":
            try:
                spec = tf.nest.map_structure(lambda t: tf.TensorSpec.from_tensor(tf.convert_to_tensor(t)), x)
                tf.function(lambda inp: self.model(inp, training=False)).get_concrete_function(spec)
                return
            except Exception as e:  # noqa  (e.g. some strategies cannot lift initializers while tracing)
                self.build_info = f"eager (trace build failed: {type(e).__name__})"
                print(f"apexnet: trace-build failed ({type(e).__name__}); falling back to an eager build")
                for m in (self.model, *self.model.submodules):
                    if hasattr(m, "_built"):
                        m._built = False
        self.model(x, training=False)
        live = {id(v) for v in self.model.variables}   # drop regularizers that point at orphaned variables
        for m in (self.model, *self.model.submodules):
            if getattr(m, "_regs", None):
                m._regs = [(v, f) for v, f in m._regs if id(v) in live]

    def _restore(self, path):
        return self._ckpt.restore(path, options=self.ckpt_opts).expect_partial()

    def get_weights(self):
        return [v.numpy() for v in self.model.variables]

    def set_weights(self, ws):
        for v, w in zip(self.model.variables, ws):
            v.assign(w)

    def save(self, path):
        assert self._ready, "nothing to save yet: call fit/evaluate or trainer.init(x) first"
        return self._ckpt.write(path, options=self.ckpt_opts)

    def load(self, path):
        """Restore full training state. Safe to call before the first batch has been seen."""
        if self._ready:
            self._restore(path)
        else:
            self._pending = path

    @_own_precision
    def init(self, x):
        self._ensure_init(x)
        return self

    @property
    def optimizers(self):
        return ([self.opt] if self.opt is not None else []) + [o for o, _ in self.extra_opts.values()]

    @property
    def main_opt(self):
        return self.optimizers[0]

    @property
    def epoch(self):
        return int(self._epoch.numpy()) if self._ready else 0

    # ------------------------------------------------------- the step logic
    def _f32(self, t):
        return tf.nest.map_structure(lambda a: tf.cast(a, tf.float32) if a.dtype.is_floating else a, t)

    def _loss(self, y, pred, w):
        out = self.loss_fn(y, pred)
        lv, lw = out if isinstance(out, (tuple, list)) else (out, tf.ones_like(out))
        lv, lw = tf.cast(lv, tf.float32), tf.cast(lw, tf.float32)
        if w is not None:
            w = tf.cast(w, tf.float32)
            while w.shape.rank < lw.shape.rank:
                w = tf.expand_dims(w, -1)
            lw = lw * w
        return lv, lw

    def _stats(self, y, pred, w, num, den):
        out = {self.loss_name: (num, den)}
        for n, m in self.metrics.items():
            out[n] = m(y, pred, w)
        return out

    def _forward_backward(self, x, y, w, ctx, tv):
        with tf.GradientTape() as tape:
            pred = self._f32(self.model(x, training=True))
            lv, lw = self._loss(y, pred, w)
            num, wsum = tf.reduce_sum(lv * lw), tf.reduce_sum(lw)
            den = ctx.all_reduce("sum", wsum)  # global normaliser => exact under any sharding
            loss = num / tf.maximum(den, 1e-12)
            if hasattr(self.model, "regularization_loss"):  # divided by replicas: grads are summed later
                loss += self.reg_weight * self.model.regularization_loss() / tf.cast(ctx.num_replicas_in_sync, tf.float32)
            scaled = loss * self.scaler.scale if self.scaler is not None else loss
        grads = tape.gradient(scaled, tv)
        grads = [tf.zeros_like(v) if g is None else tf.convert_to_tensor(g) for g, v in zip(grads, tv)]
        return grads, pred, num, wsum

    def train_step(self, batch):
        """Per-replica step. Override for GANs, custom objectives, etc. Must return stats dict."""
        x, y, w = _unpack(batch)
        ctx = tf.distribute.get_replica_context()
        tv = self.model.trainable_variables
        grads, pred, num, wsum = self._forward_backward(x, y, w, ctx, tv)
        if self.sam:  # SAM: climb to the worst-case neighbour, take the gradient THERE, then restore weights
            red = ctx.all_reduce("sum", grads)
            if self.scaler is not None:
                red = [g / self.scaler.scale for g in red]
            gn = tf.linalg.global_norm(red)
            ok = tf.math.is_finite(gn)  # never perturb with NaN/Inf: that would poison the weights
            eps = [tf.where(ok, self.sam * g / (gn + 1e-12), tf.zeros_like(g)) for g in red]
            backup = [tf.identity(v) for v in tv]
            for v, e in zip(tv, eps):
                v.assign_add(e)
            grads, _, _, _ = self._forward_backward(x, y, w, ctx, tv)
            with tf.control_dependencies(grads):
                for v, b in zip(tv, backup):
                    v.assign(b)
        self._update(grads, tv)
        return self._stats(y, pred, w, num, wsum)

    def eval_step(self, batch):
        x, y, w = _unpack(batch)
        pred = self._f32(self.model(x, training=False))
        lv, lw = self._loss(y, pred, w)
        return self._stats(y, pred, w, tf.reduce_sum(lv * lw), tf.reduce_sum(lw))

    def _update(self, grads, tv):
        """All state changes here are branch-free (masked) so they are legal in replica context."""
        ctx = tf.distribute.get_replica_context()
        grads = ctx.all_reduce("sum", grads)
        if self.scaler is not None:
            grads = [g / self.scaler.scale for g in grads]
        if self.accumulate > 1:
            k = float(self.accumulate)
            summed = [a + g for a, g in zip(self.acc, grads)]
            boundary = self.micro + 1 >= self.accumulate
            self._apply([g / k for g in summed], tv, boundary)
            # writes carry the data dependency, so nothing fires on an exhausted iterator
            writes = [a.assign(tf.where(boundary, tf.zeros_like(g), g)) for a, g in zip(self.acc, summed)]
            with tf.control_dependencies(writes):
                self.micro.assign(tf.where(boundary, tf.zeros_like(self.micro), self.micro + 1))
        else:
            self._apply(grads, tv, tf.constant(True))

    def _apply(self, grads, tv, boundary, opt=None):
        opt = opt or self.opt
        norm = tf.linalg.global_norm(grads)
        finite = tf.math.is_finite(norm)
        if self.grad_clip:
            s = tf.minimum(1.0, self.grad_clip / (norm + 1e-6))
            grads = [g * s for g in grads]
        go = boundary & (finite | (not self.skip_nonfinite))
        if self.cond_updates:
            # Single replica: tf.cond is legal around variable updates, so the update runs UNMASKED (one write per
            # variable instead of read+select+write) and is skipped entirely on non-finite / non-boundary steps.
            def do():
                opt.apply(grads, tv, None)
                if self.ema_decay:
                    n = tf.cast(opt.iterations, tf.float32)
                    d = tf.minimum(self.ema_decay, (1.0 + n) / (10.0 + n))
                    for e, v in zip(self.ema_vars, self.model.variables):
                        e.assign(d * e + (1.0 - d) * v)
                return tf.constant(0)
            tf.cond(go, do, lambda: tf.constant(0))
        else:
            # Mirrored / TPU: variable updates inside tf.cond need merge_call (illegal), so every write is masked.
            opt.apply(grads, tv, go)
            if self.ema_decay:
                n = tf.cast(opt.iterations, tf.float32)
                d = tf.minimum(self.ema_decay, (1.0 + n) / (10.0 + n))
                for e, v in zip(self.ema_vars, self.model.variables):
                    e.assign(tf.where(go, d * e + (1.0 - d) * v, e))
        self.skipped.assign_add(tf.cast(boundary & ~finite, tf.int64))
        self.last_norm.assign(tf.where(boundary, norm, self.last_norm))
        if self.scaler is not None:
            self.scaler.update(finite | ~boundary)

    @_own_precision
    def lr_find(self, data, y=None, *, batch_size=None, start=1e-7, end=1.0, steps=100, beta=0.9,
                divergence=4.0):
        """Leslie Smith's LR range test. Trains ``steps`` steps with an exponentially growing LR, then restores
        ALL training state (weights, optimizer slots, counters). Returns {lrs, losses, suggested}; ``suggested`` is
        the LR at the smoothed-loss minimum divided by 10."""
        import shutil, tempfile
        ds = self._prepare(data, y, batch_size, True)
        self._ensure_init(_unpack(next(iter(ds)))[0])
        tmp = tempfile.mkdtemp()
        ck = self.save(os.path.join(tmp, "s"))
        probe = tf.Variable(start, dtype=tf.float32, trainable=False)
        saved = [(o, o.lr, float(o.lr_scale.numpy())) for o in self.optimizers]
        for o in self.optimizers:
            o.lr, _ = (lambda step: probe), o.lr_scale.assign(1.0)
        loop = self._make_loop(self.train_step, self.train_meter)  # fresh trace that reads `probe`
        dist = self._dist(ds)
        it, lrs, losses, avg, best = iter(dist), [], [], 0.0, float("inf")
        try:
            for i in range(steps):
                lr = start * (end / start) ** (i / max(1, steps - 1))
                probe.assign(lr)
                self.train_meter.reset()
                try:
                    loop(it, tf.constant(1))
                except tf.errors.OutOfRangeError:
                    it = iter(dist)
                    loop(it, tf.constant(1))
                loss = self.train_meter.result(self.metrics, self.loss_name)[self.loss_name]
                if not np.isfinite(loss):
                    break
                avg = beta * avg + (1 - beta) * loss
                sm = avg / (1 - beta ** (i + 1))
                lrs.append(lr); losses.append(sm)
                best = min(best, sm)
                if i > 5 and sm > divergence * best:
                    break
        finally:
            for o, lr0, sc in saved:
                o.lr = lr0
            self._restore(ck)
            for o, _, sc in saved:
                o.lr_scale.assign(sc)
            self.train_meter.reset()
            shutil.rmtree(tmp, ignore_errors=True)
        suggested = lrs[int(np.argmin(losses))] / 10.0 if losses else start
        return {"lrs": lrs, "losses": losses, "suggested": suggested}

    def apply_gradients(self, opt, grads, variables):
        """For custom ``train_step``s: all-reduce -> (unscale) -> clip -> NaN-guard -> ``opt.apply``.
        ``opt`` may be an optimizer or the name of an ``extra_optimizers`` entry. No accumulation."""
        opt = self.extra_opts[opt][0] if isinstance(opt, str) else opt
        ctx = tf.distribute.get_replica_context()
        grads = [tf.zeros_like(v) if g is None else tf.convert_to_tensor(g) for g, v in zip(grads, variables)]
        grads = ctx.all_reduce("sum", grads)
        if self.scaler is not None:
            grads = [g / self.scaler.scale for g in grads]
        self._apply(grads, variables, tf.constant(True), opt)

    # ------------------------------------------------------------- the loops
    def _make_loop(self, step, meter):
        def one(batch):
            out = self.strategy.run(step, args=(batch,))
            meter.update(tf.nest.map_structure(lambda t: self.strategy.reduce(SUM, t, axis=None), out))

        @tf.function
        def loop(it, n):
            for _ in tf.range(n):
                one(next(it))
        return loop

    def _known_cardinality(self, ds):
        """-> (ds, card). card: int >= 0 (exact #batches), None (infinite), -1 (unknown, single replica only).
        Multi-replica or static-shape runs must know the exact count: running a step into the end of a distributed
        iterator can execute part of it on a phantom batch (corrupting the metric accumulators)."""
        c = int(ds.cardinality().numpy())
        if c == tf.data.INFINITE_CARDINALITY:
            return ds, None
        if c >= 0:
            return ds, c
        if self.strategy.num_replicas_in_sync > 1 or self.static_batches:
            c = int(ds.reduce(tf.constant(0, tf.int64), lambda n, _: n + 1).numpy())  # one counting pass
            return ds.apply(tf.data.experimental.assert_cardinality(c)), c
        return ds, -1

    def _drive(self, loop, feed, steps, cb=None):
        done, empty = 0, 0
        while steps is None or done < steps:
            n = self.steps_per_call if steps is None else min(self.steps_per_call, steps - done)
            exact = feed.card is not None and feed.card >= 0
            if exact:  # never call into the end of the data
                if feed.remaining == 0:
                    if steps is None or feed.card == 0:
                        break
                    feed.reset()  # finite data + steps_per_epoch: wrap around
                n = min(n, feed.remaining)
            try:
                loop(feed.it, tf.constant(n))
            except (tf.errors.OutOfRangeError, StopIteration):  # only for unknown cardinality
                if steps is None:
                    break
                feed.reset()
                empty += 1
                if empty > 1:
                    break
                continue
            if exact:
                feed.remaining -= n
            done += n
            if cb is not None:
                cb.on_step_end(self, done)

    @staticmethod
    def _norm_arrays(data, y, w=None):
        if y is None and isinstance(data, (tuple, list)) and len(data) in (2, 3):
            data, y, w = (*data, None)[:3] if len(data) == 2 else data
        return data, y, w

    def _prepare(self, data, y, batch_size, shuffle, sample_weight=None, class_weight=None):  # shuffle==training
        if _is_dataset(data):
            return D.with_class_weight(data, class_weight) if class_weight else data
        data, y, w = self._norm_arrays(data, y, sample_weight)
        if class_weight:
            cw = D.class_weight_vector(class_weight, y)
            w = cw if w is None else np.asarray(w, np.float32) * cw
        return D.from_arrays(data, y, w, batch_size=batch_size or 32, shuffle=shuffle,
                             drop_remainder=bool(self.static_batches and shuffle))

    def _check_batch(self, sample):
        n = self.strategy.num_replicas_in_sync
        b = int(tf.nest.flatten(sample)[0].shape[0])
        if n > 1 and b % n:
            raise ValueError(f"global batch size {b} must be divisible by the number of replicas ({n}); "
                             f"use batch_size={max(n, b // n * n)} (or hardware.global_batch(per_replica_batch))")
        return b

    def _static_ds(self, ds, pad, with_weights=True):
        """Static batch shapes (TPU/XLA). Training (pad=False): batches that are already static pass through untouched
        (cardinality preserved); otherwise re-batch and drop the ragged tail. Eval/predict (pad=True): append zero
        examples (weight 0 where weights exist) so every batch is full; results are unaffected."""
        leaf0 = tf.nest.flatten(ds.element_spec)[0]
        if not pad and leaf0.shape.rank and leaf0.shape[0] is not None:
            return ds
        sample = next(iter(ds))
        B = int(tf.nest.flatten(sample)[0].shape[0])
        n = int(ds.reduce(tf.constant(0, tf.int64), lambda c, b: c + tf.cast(tf.shape(tf.nest.flatten(b)[0])[0], tf.int64)).numpy())
        u = ds.unbatch()
        if not pad:
            return u.batch(B, drop_remainder=True).apply(tf.data.experimental.assert_cardinality(n // B))
        if with_weights:  # (x, y[, w]) -> (x, y, w)
            u = u.map(lambda *e: (e[0], e[1], e[2] if len(e) > 2 else tf.constant(1.0)))
        spec = u.element_spec
        if any(d is None for t in tf.nest.flatten(spec) for d in t.shape):
            raise ValueError("static batches need fixed per-example shapes; pad variable-length data first "
                             "(e.g. data.bucket_by_length(...) or padded_batch to a constant length)")
        zeros = tf.nest.map_structure(lambda t: tf.zeros(t.shape, t.dtype) if t.dtype != tf.string else tf.fill(t.shape, ""), spec)
        padding = tf.data.Dataset.from_tensors(zeros).repeat(max(B - 1, 0))
        return u.concatenate(padding).batch(B, drop_remainder=True).apply(
            tf.data.experimental.assert_cardinality(-(-n // B)))

    def _dist(self, ds):
        return self.strategy.experimental_distribute_dataset(ds)

    # ------------------------------------------------------------------- API
    @_own_precision
    def fit(self, data, y=None, *, epochs=1, batch_size=None, val_data=None, validation_split=0.0,
            class_weight=None, sample_weight=None, shuffle=True, callbacks=(), steps_per_epoch=None,
            verbose=1, log_every=0):
        """Train until ``epochs`` TOTAL epochs are done (resumes after ``load``).
        ``validation_split`` (arrays only) holds out the LAST fraction; ``class_weight``: {class: weight}
        (give every class); ``sample_weight``: per-example weights for arrays. Losses are normalised by the
        weight sum (a weighted mean), not by the example count."""
        if validation_split and val_data is None:
            if _is_dataset(data):
                raise ValueError("validation_split needs array inputs; pass val_data for datasets")
            data, y, sample_weight = self._norm_arrays(data, y, sample_weight)
            n = len(tf.nest.flatten(data)[0])
            k = int(n * validation_split)
            cut = lambda a, lo, hi: tf.nest.map_structure(lambda t: t[lo:hi], a)
            val_data = (cut(data, n - k, n), cut(y, n - k, n))
            data, y = cut(data, 0, n - k), cut(y, 0, n - k)
            sample_weight = None if sample_weight is None else sample_weight[: n - k]
        ds = self._prepare(data, y, batch_size, shuffle, sample_weight, class_weight)
        if self.static_batches:
            ds = self._static_ds(ds, pad=False)
        sample = next(iter(ds))
        self._check_batch(sample)
        self._ensure_init(_unpack(sample)[0])
        self._verify_xla("train", sample)
        cbs = CallbackList(([Logger(log_every)] if verbose else []) + list(callbacks))
        ds, card = self._known_cardinality(ds)
        dist, self.stop, self.target_epochs = self._dist(ds), False, epochs
        persistent = _Feed(lambda: iter(dist), card) if steps_per_epoch else None
        cbs.on_fit_start(self)
        for ep in range(self.epoch, epochs):
            cbs.on_epoch_start(self, ep)
            self.train_meter.reset()
            feed = persistent or _Feed(lambda: iter(dist), card)
            self._drive(self._train_loop, feed, steps_per_epoch, cbs)
            logs = self.train_meter.result(self.metrics, self.loss_name)
            logs["lr"] = float(self.main_opt.current_lr().numpy())
            logs["grad_norm"] = float(self.last_norm.numpy())
            if int(self.skipped.numpy()):
                logs["skipped"] = float(self.skipped.numpy())
            if val_data is not None:
                logs.update({"val_" + k: v for k, v in self.evaluate(val_data, batch_size=batch_size).items()})
            self._epoch.assign(ep + 1)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)
            cbs.on_epoch_end(self, ep, logs)
            if self.stop:
                break
        cbs.on_fit_end(self)
        return self.history

    @contextlib.contextmanager
    def ema_weights(self):
        """Temporarily swap in EMA weights (used automatically by evaluate/predict)."""
        vs = self.model.variables
        backup = [v.numpy() for v in vs]
        for v, e in zip(vs, self.ema_vars):
            v.assign(e.numpy())
        try:
            yield
        finally:
            for v, b in zip(vs, backup):
                v.assign(b)

    def _maybe_ema(self, use_ema):
        return self.ema_weights() if (use_ema and self.ema_decay and self._ready) else contextlib.nullcontext()

    @_own_precision
    def evaluate(self, data, y=None, *, batch_size=None, use_ema=True):
        ds = self._prepare(data, y, batch_size, shuffle=False)
        if self.static_batches:
            ds = self._static_ds(ds, pad=True)
        sample = next(iter(ds))
        self._check_batch(sample)
        self._ensure_init(_unpack(sample)[0])
        self._verify_xla("eval", sample)
        self.eval_meter.reset()
        ds, card = self._known_cardinality(ds)
        dist = self._dist(ds)
        with self._maybe_ema(use_ema):
            self._drive(self._eval_loop, _Feed(lambda: iter(dist), card), None)
        return self.eval_meter.result(self.metrics, self.loss_name)

    @_own_precision
    def predict(self, x, batch_size=64, use_ema=True):
        """Batched inference on the trainer's strategy (all replicas / the TPU). Output rows follow input order."""
        ds = x if _is_dataset(x) else D.from_arrays(x, batch_size=batch_size)
        ds = ds.map(lambda *b: b[0])
        n_valid = None
        if self.static_batches:
            n_valid = int(ds.reduce(tf.constant(0, tf.int64), lambda c, b: c + tf.cast(tf.shape(tf.nest.flatten(b)[0])[0], tf.int64)).numpy())
            ds = self._static_ds(ds, pad=True, with_weights=False)
        sample = next(iter(ds))
        self._check_batch(sample)
        self._ensure_init(sample)
        fwd = lambda b: self._f32(self.model(b, training=False))

        @tf.function
        def run(b):
            return self.strategy.run(fwd, args=(b,))
        outs = []
        with self._maybe_ema(use_ema):
            for b in self._dist(ds):
                parts = self.strategy.experimental_local_results(run(b))
                outs.append(tf.nest.map_structure(lambda *ts: tf.concat(ts, 0).numpy(), *parts))
        res = tf.nest.map_structure(lambda *a: np.concatenate(a), *outs)
        return res if n_valid is None else tf.nest.map_structure(lambda a: a[:n_valid], res)

    def summary(self):
        return self.model.summary()

    @contextlib.contextmanager
    def _export_ctx(self, precision):
        from .core import compute_dtype
        old = {tf.float32: "float32", tf.bfloat16: "bfloat16", tf.float16: "float16"}[compute_dtype()]
        set_precision(precision)
        try:
            with self._maybe_ema(True):
                yield
        finally:
            set_precision(old)

    def _serve_module(self, input_signature):
        m = tf.Module()
        m.model = self.model
        m.serve = tf.function(lambda x: self._f32(self.model(x, training=False)), input_signature=input_signature)
        return m

    def export(self, path, input_signature, precision="float32"):
        """SavedModel with a `serve` signature, e.g. [tf.TensorSpec([None, 20], tf.float32)].
        Exports with EMA weights (if enabled) in float32 by default."""
        with self._export_ctx(precision):
            m = self._serve_module(input_signature)
            opts = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost") if self.is_tpu else None
            tf.saved_model.save(m, path, signatures={"serving_default": m.serve}, options=opts)
        return path

    def export_tflite(self, path, input_signature, quantize=None):
        """TFLite flatbuffer. quantize: None | 'dynamic' (int8 weights) | 'float16'."""
        with self._export_ctx("float32"):
            m = self._serve_module(input_signature)
            conv = tf.lite.TFLiteConverter.from_concrete_functions([m.serve.get_concrete_function()], m)
            if quantize:
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
                if quantize == "float16":
                    conv.target_spec.supported_types = [tf.float16]
            blob = conv.convert()
        with open(path, "wb") as f:
            f.write(blob)
        return path
