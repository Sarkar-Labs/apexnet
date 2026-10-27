"""Core building blocks: precision policy, initializers, Module, Sequential.

Design notes
------------
* ``Module`` is a plain ``tf.Module``: variables, checkpoints, SavedModel and
  ``tf.function`` all work natively, no private Keras machinery.
* Shapes are inferred lazily on first call (``Dense(64)`` needs no input size).
* ``training`` is always an explicit argument. There is no hidden global state
  except the (optional) mixed-precision policy.
"""
import functools
import inspect
import json
import math
import os
import re
import weakref
import numpy as np
import tensorflow as tf

# ------------------------------------------------------------ config registry
_REGISTRY = {}        # class name -> Module subclass
_FN_REGISTRY = {}     # function name -> function (for serialisable callables)
_OBJECTS = {}         # config-object kind -> factory (regularizers, initializers)
_CONFIGS = weakref.WeakKeyDictionary()   # module -> (class name, captured ctor args)


def register(obj=None, name=None):
    """Make a custom Module class (or a plain function used as a layer argument) serialisable by name."""
    def deco(o):
        (_REGISTRY if inspect.isclass(o) else _FN_REGISTRY)[name or o.__name__] = o
        return o
    return deco(obj) if obj is not None else deco


def register_object(kind, factory):
    _OBJECTS[kind] = factory


class Configurable:
    """Tiny callable object that knows how to serialise itself (regularizers, parametrised initializers)."""
    def __init__(self, kind, fn, **kw):
        self.kind, self.fn, self.kw = kind, fn, kw

    @property
    def apexnet_config(self):
        return self.kind, self.kw

    def __call__(self, *a, **k):
        return self.fn(*a, **k)



# ----------------------------------------------------------------- precision
_DTYPES = {"float32": tf.float32, "bfloat16": tf.bfloat16, "float16": tf.float16}


class _Policy:
    compute_dtype = tf.float32


def set_precision(name):
    """'float32' | 'bfloat16' | 'float16'. Variables stay float32; math runs in `name`."""
    _Policy.compute_dtype = _DTYPES[name]


def compute_dtype():
    return _Policy.compute_dtype


# -------------------------------------------------------------- initializers
def _fans(shape):
    if len(shape) == 1:
        return shape[0], shape[0]
    rf = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    return shape[-2] * rf, shape[-1] * rf


def glorot_uniform(shape, dtype=tf.float32):
    fi, fo = _fans(shape)
    lim = math.sqrt(6.0 / (fi + fo))
    return tf.random.uniform(shape, -lim, lim, dtype)


def he_normal(shape, dtype=tf.float32):
    fi, _ = _fans(shape)
    return tf.random.normal(shape, 0.0, math.sqrt(2.0 / fi), dtype)


def zeros(shape, dtype=tf.float32):
    return tf.zeros(shape, dtype)


def ones(shape, dtype=tf.float32):
    return tf.ones(shape, dtype)


def normal(std=0.02):
    return Configurable("normal", lambda shape, dtype=tf.float32: tf.random.normal(shape, 0.0, std, dtype), std=std)


def lecun_normal(shape, dtype=tf.float32):
    fi, _ = _fans(shape)
    return tf.random.normal(shape, 0.0, math.sqrt(1.0 / fi), dtype)


def he_uniform(shape, dtype=tf.float32):
    fi, _ = _fans(shape)
    lim = math.sqrt(6.0 / fi)
    return tf.random.uniform(shape, -lim, lim, dtype)


def glorot_normal(shape, dtype=tf.float32):
    fi, fo = _fans(shape)
    return tf.random.normal(shape, 0.0, math.sqrt(2.0 / (fi + fo)), dtype)


def orthogonal(shape, dtype=tf.float32):
    rows = int(np.prod(shape[:-1]))
    cols = int(shape[-1])
    a = tf.random.normal((max(rows, cols), min(rows, cols)), 0.0, 1.0, tf.float32)
    q, r = tf.linalg.qr(a)
    q = q * tf.sign(tf.linalg.diag_part(r))
    if rows < cols:
        q = tf.transpose(q)
    return tf.cast(tf.reshape(q[:rows, :cols], shape), dtype)


def truncated_normal(std=0.02):
    return Configurable("truncated_normal", lambda shape, dtype=tf.float32: tf.random.truncated_normal(
        shape, 0.0, std, dtype), std=std)


def uniform(limit=0.05):
    return Configurable("uniform", lambda shape, dtype=tf.float32: tf.random.uniform(
        shape, -limit, limit, dtype), limit=limit)


def constant(value):
    return Configurable("constant", lambda shape, dtype=tf.float32: tf.fill(shape, tf.cast(value, dtype)),
                        value=value)


_INITS = dict(glorot_uniform=glorot_uniform, glorot_normal=glorot_normal, he_normal=he_normal,
              he_uniform=he_uniform, lecun_normal=lecun_normal, orthogonal=orthogonal,
              zeros=zeros, ones=ones)


for _k, _f in (("normal", lambda std: normal(std)), ("truncated_normal", lambda std: truncated_normal(std)),
               ("uniform", lambda limit: uniform(limit)), ("constant", lambda value: constant(value))):
    _OBJECTS[_k] = _f


def get_init(spec):
    if callable(spec):
        return spec
    if isinstance(spec, (int, float)):
        return normal(float(spec))
    return _INITS[spec]



# ------------------------------------------------------------ config registry
def _fn_name(fn):
    from .layers import ACTIVATIONS
    for k, v in ACTIVATIONS.items():
        if v is fn and k is not None:
            return k
    for k, v in _FN_REGISTRY.items():
        if v is fn:
            return k
    return None


def _ser(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, Module):
        return {"__module__": v.get_config()}
    if isinstance(v, tuple):
        return {"__tuple__": [_ser(x) for x in v]}
    if isinstance(v, list):
        return [_ser(x) for x in v]
    if isinstance(v, dict):
        return {"__dict__": {str(k): _ser(x) for k, x in v.items()}}
    if isinstance(v, tf.DType):
        return {"__dtype__": v.name}
    if hasattr(v, "apexnet_config"):
        kind, kw = v.apexnet_config
        return {"__obj__": kind, "kw": _ser(kw)}
    if callable(v):
        n = _fn_name(v)
        if n is not None:
            return {"__fn__": n}
        raise ValueError(f"cannot serialise callable {v!r}: use a named activation / regularizer, or decorate "
                         f"the function with @apexnet.register")
    raise TypeError(f"cannot serialise {type(v).__name__}")


def _de(v):
    if isinstance(v, list):
        return [_de(x) for x in v]
    if isinstance(v, dict):
        if "__module__" in v:
            return from_config(v["__module__"])
        if "__tuple__" in v:
            return tuple(_de(x) for x in v["__tuple__"])
        if "__dict__" in v:
            return {k: _de(x) for k, x in v["__dict__"].items()}
        if "__dtype__" in v:
            return tf.as_dtype(v["__dtype__"])
        if "__obj__" in v:
            return _OBJECTS[v["__obj__"]](**_de(v["kw"]))
        if "__fn__" in v:
            from .layers import ACTIVATIONS
            n = v["__fn__"]
            return _FN_REGISTRY[n] if n in _FN_REGISTRY else ACTIVATIONS[n]
    return v


def _capture(orig, self, a, k):
    sig = inspect.signature(orig)
    bound = sig.bind(self, *a, **k)
    out, order = {}, []
    for name, val in list(bound.arguments.items())[1:]:
        kind = sig.parameters[name].kind
        order.append([name, kind.name])
        out[name] = list(val) if kind is inspect.Parameter.VAR_POSITIONAL else val
    return {"args": out, "order": order}


def from_config(cfg):
    """Rebuild a Module from ``module.get_config()`` (weights are not part of the config)."""
    cls = _REGISTRY[cfg["class"]]
    args = {k: _de(v) for k, v in cfg["args"].items()}
    sig = inspect.signature(cls.__init__)
    pos, kw, seen_var = [], {}, False
    for name, p in list(sig.parameters.items())[1:]:
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            if name in args:
                pos.append(args[name])
            elif any(q.kind is q.VAR_POSITIONAL for q in sig.parameters.values()):
                pos.append(p.default)
        elif p.kind is p.VAR_POSITIONAL:
            pos.extend(args.get(name, []))
        elif p.kind is p.KEYWORD_ONLY and name in args:
            kw[name] = args[name]
        elif p.kind is p.VAR_KEYWORD:
            kw.update(args.get(name, {}))
    # keyword-style rebuild when there is no *args (safest for defaults)
    if not any(q.kind is q.VAR_POSITIONAL for q in sig.parameters.values()):
        pos = []
        kw.update({n: args[n] for n, p in list(sig.parameters.items())[1:]
                   if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and n in args})
    m = cls(*pos, **kw)
    if cfg.get("state"):
        m.set_state(cfg["state"])
    return m


class _Trace:
    """Active during Module.summary(example): records every module call with nesting depth + output."""
    active = None

    def __init__(self):
        self.rows, self.depth = [], 0

# -------------------------------------------------------------------- Module
class Module(tf.Module):
    """Base class for all layers and models. Override ``build`` and ``forward``."""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        _REGISTRY[cls.__name__] = cls
        orig = cls.__init__

        @functools.wraps(orig)
        def wrapped(self, *a, **k):
            if self not in _CONFIGS:  # outermost constructor call wins (the most-derived class)
                _CONFIGS[self] = (type(self).__name__, _capture(orig, self, a, k))
            orig(self, *a, **k)
        cls.__init__ = wrapped

    def __init__(self, name=None):
        if name is None:  # snake_case class name; tf name scopes may not start with "_"
            name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(self).__name__).lower().lstrip("_") or "module"
        super().__init__(name=name)
        self._built = False
        self._frozen = False
        self._regs = []

    # -- to override
    def build(self, x):
        """Create variables. ``x`` is the first input (use ``x.shape``)."""

    def forward(self, x, *args, training=False, **kwargs):
        raise NotImplementedError

    # -- machinery
    def __call__(self, x, *args, training=False, **kwargs):
        with self.name_scope:
            if not self._built:
                self.build(x)
                self._built = True
            tr = _Trace.active
            if tr is None:
                return self.forward(x, *args, training=training, **kwargs)
            row = [tr.depth, self, None]
            tr.rows.append(row)
            tr.depth += 1
            try:
                row[2] = self.forward(x, *args, training=training, **kwargs)
                return row[2]
            finally:
                tr.depth -= 1

    # ---- serialisation
    def get_config(self):
        """JSON-able constructor description. Lambdas / Wrap'd Keras layers are not serialisable."""
        if self not in _CONFIGS:
            raise ValueError(f"{type(self).__name__} has no recorded constructor config")
        name, cap = _CONFIGS[self]
        cfg = {"class": name, "args": {k: _ser(v) for k, v in cap["args"].items()}}
        st = self.get_state()
        if st:
            cfg["state"] = st
        return cfg

    def get_state(self):
        """Override to persist non-variable state (e.g. a vocabulary) as a JSON-able dict."""
        return None

    def set_state(self, state):
        pass

    def param(self, name, shape, init="glorot_uniform", trainable=True, regularizer=None):
        """Create a float32 variable. ``regularizer``: callable var -> scalar (see apexnet.regularizers)."""
        fn = get_init(init)
        shape = tuple(int(s) for s in shape)
        v = tf.Variable(
            lambda: fn(shape, tf.float32), name=name, trainable=trainable,
            dtype=tf.float32, shape=shape,
            # mean-aggregation makes updates legal inside tf.distribute replicas
            aggregation=tf.VariableAggregation.MEAN)
        if regularizer is not None:
            self._regs.append((v, regularizer))
        return v

    # ---- freezing / transfer learning (set BEFORE creating the Trainer)
    def freeze(self):
        """Exclude this module (and everything below it) from training; BatchNorm runs in eval mode."""
        for m in (self, *self.submodules):
            m._frozen = True
        return self

    def unfreeze(self):
        for m in (self, *self.submodules):
            m._frozen = False
        return self

    @property
    def trainable_variables(self):
        out, seen = [], set()

        def walk(m):
            if getattr(m, "_frozen", False):
                return
            for v in m._flatten(recursive=False, predicate=lambda t: isinstance(t, tf.Variable)):
                if v.trainable and id(v) not in seen:
                    seen.add(id(v))
                    out.append(v)
            for c in m._flatten(recursive=False, predicate=lambda t: isinstance(t, tf.Module)):
                walk(c)
        walk(self)
        return tuple(out)

    def regularization_loss(self):
        """Sum of all registered weight regularizers (frozen modules excluded)."""
        total = tf.constant(0.0)
        for m in (self, *self.submodules):
            if isinstance(m, Module) and not m._frozen:
                for v, fn in m._regs:
                    total += tf.cast(fn(v), tf.float32)
        return total

    @staticmethod
    def cast(x):
        x = tf.convert_to_tensor(x)
        return tf.cast(x, compute_dtype()) if x.dtype.is_floating else x

    def init(self, x):
        """Build all variables eagerly from an example input. Returns self."""
        self(x, training=False)
        return self

    def count_params(self, trainable_only=False):
        vs = self.trainable_variables if trainable_only else self.variables
        return int(sum(np.prod(v.shape) for v in vs))

    def summary(self, example=None):
        """Parameter table. With ``example`` (an input batch) it also shows each layer's output shape,
        nested by call depth, like ``keras.Model.summary``."""
        def own_params(m):
            own = list(m._flatten(recursive=False, predicate=lambda t: isinstance(t, tf.Variable)))
            return int(sum(np.prod(v.shape) for v in own)), ", ".join(str(tuple(v.shape.as_list())) for v in own)

        def shp(o):
            return ", ".join(str(tuple(t.shape.as_list())) for t in tf.nest.flatten(o) if hasattr(t, "shape"))
        rows, seen = [], {}
        if example is not None:
            self(example, training=False)  # make sure everything is built
            tr = _Trace()
            _Trace.active = tr
            try:
                self(example, training=False)
            finally:
                _Trace.active = None
            for depth, m, out in tr.rows:
                n, _ = own_params(m)
                rows.append(("  " * depth + m.name, type(m).__name__, shp(out), n))
            hdr = ("layer", "type", "output shape", "params")
        else:
            for m in (self, *self.submodules):
                if isinstance(m, Module):
                    n, sh = own_params(m)
                    if n:
                        seen[m.name] = seen.get(m.name, 0) + 1
                        rows.append((m.name if seen[m.name] == 1 else f"{m.name}_{seen[m.name] - 1}",
                                     type(m).__name__, sh, n))
            hdr = ("name", "type", "shapes", "params")
        w0 = max([len(r[0]) for r in rows] + [len(hdr[0])])
        w1 = max([len(r[1]) for r in rows] + [4])
        w2 = min(max([len(r[2]) for r in rows] + [6]), 48)
        lines = [f"{hdr[0]:<{w0}}  {hdr[1]:<{w1}}  {hdr[2]:<{w2}}  {hdr[3]:>9}"]
        lines += [f"{a:<{w0}}  {b:<{w1}}  {c[:w2]:<{w2}}  {n:>9,}" for a, b, c, n in rows]
        lines.append(f"total params: {self.count_params():,} (trainable {self.count_params(True):,})")
        text = "\n".join(lines)
        print(text)
        return text


class Sequential(Module):
    """Chain of Modules and/or plain callables (e.g. ``tf.nn.relu``)."""

    def __init__(self, *layers, name=None):
        super().__init__(name=name)
        self.layers = list(layers)

    def forward(self, x, training=False, **kw):
        for layer in self.layers:
            x = layer(x, training=training) if isinstance(layer, Module) else layer(x)
        return x


# -------------------------------------------------------------- save / load whole models
def _spec_of(example):
    """Nested structure of arrays -> JSON spec (batch dim recorded as 1)."""
    if isinstance(example, tuple):
        return {"__tuple__": [_spec_of(x) for x in example]}
    if isinstance(example, list):
        return [_spec_of(x) for x in example]
    if isinstance(example, dict):
        return {"__dict__": {k: _spec_of(x) for k, x in example.items()}}
    t = tf.convert_to_tensor(example)
    return {"__spec__": [1] + [int(d) for d in t.shape[1:]], "dtype": t.dtype.name}


def _zeros_from_spec(spec):
    def mk(sp):
        shape, dt = sp["__spec__"], tf.as_dtype(sp["dtype"])
        return tf.fill(shape, "") if dt == tf.string else tf.zeros(shape, dt)

    def walk(v):
        if isinstance(v, dict) and "__spec__" in v:
            return mk(v)
        if isinstance(v, dict) and "__tuple__" in v:
            return tuple(walk(x) for x in v["__tuple__"])
        if isinstance(v, dict) and "__dict__" in v:
            return {k: walk(x) for k, x in v["__dict__"].items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v
    return walk(spec)


def save_model(model, path, example=None, options=None):
    """Writes ``path/config.json`` (architecture) + ``path/weights.*`` (all variables). ``example``: an input
    batch (array or nested structure); required if the model is not built yet or takes non-trivial inputs.
    ``options``: tf.train.CheckpointOptions (``hardware.checkpoint_options(strategy)`` on TPU)."""
    os.makedirs(path, exist_ok=True)
    if example is not None:
        model(example, training=False)
    elif not model.variables:
        raise ValueError("model has no variables yet: pass example= so its input spec can be recorded")
    meta = {"format": 1, "model": model.get_config(), "input": _spec_of(example) if example is not None else None}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(meta, f, indent=1)
    tf.train.Checkpoint(model=model).write(os.path.join(path, "weights"), options=options)
    return path


def load_model(path, options=None):
    """Rebuild the architecture from config.json, build it from the stored input spec, restore the weights."""
    with open(os.path.join(path, "config.json")) as f:
        meta = json.load(f)
    model = from_config(meta["model"])
    if meta["input"] is not None:
        model(_zeros_from_spec(meta["input"]), training=False)
    tf.train.Checkpoint(model=model).restore(os.path.join(path, "weights"), options=options).expect_partial()
    return model
