"""Use JAX code inside ApexNet: ``an.Jax(fn, params)``.

``fn(params, x)`` (or ``fn(params, x, training=...)`` / ``fn(params, x, key)`` with ``rng=True``) is a *pure* JAX
function; ``params`` is any pytree of float arrays. ApexNet owns the parameters as ordinary ``tf.Variable`` s (so the
ApexNet optimizers, EMA, checkpointing, mixed precision and distribution all just work) and the JAX computation is
embedded in the TensorFlow graph through ``jax2tf``: it runs inside ``tf.function``, on GPU/TPU through XLA, with exact
JAX gradients (``jax.vjp``).

    def mlp(p, x): return jnp.tanh(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]
    layer = an.Jax(mlp, {"w1": w1, "b1": b1, "w2": w2, "b2": b2})
    model = an.Sequential(an.Dense(32, "relu"), layer)

Constraints: static batch shapes (the Trainer enables ``static_batches`` automatically: ragged training tail dropped,
eval/predict zero-padded exactly); pure functions only (no mutable state such as BatchNorm statistics); not
JSON-serialisable (a function cannot be stored) but checkpoints / SavedModel work.
Flax: ``an.Jax(lambda p, x: module.apply({"params": p}, x), module.init(key, x)["params"])`` (untested here).
"""
import inspect
import numpy as np
import tensorflow as tf
from .core import Module


def _import():
    try:
        import jax
        from jax.experimental import jax2tf
        return jax, jax2tf
    except ImportError as e:  # pragma: no cover
        raise ImportError("an.Jax needs `pip install jax jaxlib` (the jax2tf bridge ships with jax)") from e


class Jax(Module):
    requires_static_shapes = True  # read by Trainer: switches on static batches

    def __init__(self, fn, params, rng=False, name=None):
        super().__init__(name)
        jax, _ = _import()
        leaves, self.treedef = jax.tree_util.tree_flatten(params)
        self._init_vals = []
        for l in leaves:
            a = np.asarray(l)
            if a.dtype.kind != "f":
                raise TypeError(f"an.Jax parameters must be floating arrays, got {a.dtype}")
            self._init_vals.append(a.astype(np.float32))
        self.fn, self.rng = fn, rng
        self.vs = []
        self._conv = {}
        try:
            self._takes_training = "training" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            self._takes_training = False

    def build(self, x):
        self.vs = [self.param(f"p{i}", a.shape, init=(lambda shape, dtype=tf.float32, a=a: tf.constant(a, dtype)))
                   for i, a in enumerate(self._init_vals)]

    def params(self):
        """Current parameters as a JAX pytree of numpy arrays (e.g. to hand back to pure JAX code)."""
        jax, _ = _import()
        return jax.tree_util.tree_unflatten(self.treedef, [v.numpy() for v in self.vs])

    def _converted(self, training):
        key = (bool(training), self.rng)
        if key not in self._conv:
            jax, jax2tf = _import()
            fn, takes = self.fn, self._takes_training
            if self.rng:
                f = (lambda p, x, k: fn(p, x, k, training=training)) if takes else (lambda p, x, k: fn(p, x, k))
            else:
                f = (lambda p, x: fn(p, x, training=training)) if takes else fn
            # autograph must not try to convert JAX internals (noisy warnings, and slow)
            self._conv[key] = tf.autograph.experimental.do_not_convert(jax2tf.convert(f, with_gradient=True))
        return self._conv[key]

    def forward(self, x, training=False):
        jax, _ = _import()
        x = tf.convert_to_tensor(x)
        if x.shape.rank is None or x.shape[0] is None:
            raise ValueError("an.Jax needs a static batch dimension (jax2tf cannot export dynamic batch shapes with "
                             "this TF/JAX pair). Use Trainer(static_batches=True) / drop_remainder=True, or pass a "
                             "fixed-size batch.")
        p = jax.tree_util.tree_unflatten(self.treedef, [v.read_value() for v in self.vs])
        args = [p, x]
        if self.rng:
            args.append(tf.cast(tf.random.uniform([2], 0, 2 ** 31 - 1, tf.int32), tf.uint32) if training
                        else tf.zeros([2], tf.uint32))
        return self._converted(training)(*args)
