"""Escape hatches: use any Keras layer, tf.Module or SavedModel inside ApexNet (and vice-versa)."""
import inspect
import tensorflow as tf
from .core import Module


def _tf_vars(obj):
    """Collect real tf.Variables from tf.Module / keras layer / loaded SavedModel."""
    out = []
    for v in getattr(obj, "variables", []):
        raw = getattr(v, "value", None)
        out.append(raw if isinstance(raw, tf.Variable) else v)
    return [v for v in out if isinstance(v, tf.Variable)]


class Wrap(Module):
    """Wrap ANY callable that owns variables: ``tf.keras.layers.*``, ``tf.Module``, ``tf.saved_model.load(...)``,
    ``hub.KerasLayer`` ... Its variables become part of the ApexNet model (optimised, checkpointed, EMA'd).
    ``trainable=False`` freezes them. The ``training`` flag is forwarded when the callable accepts it."""

    def __init__(self, obj, trainable=True, name=None):
        super().__init__(name)
        self.obj, self.keep_trainable = obj, trainable
        self._vars = []
        self._takes_training = None

    def build(self, x):
        if getattr(self.obj, "built", True) is False:
            self.obj(x)
        self._vars = _tf_vars(self.obj)
        if not self.keep_trainable:
            self._frozen = True

    def forward(self, x, *a, training=False, **kw):
        if self._takes_training is None:
            try:
                sig = inspect.signature(self.obj.__call__ if not hasattr(self.obj, "call") else self.obj.call)
                self._takes_training = "training" in sig.parameters or any(
                    p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            except (TypeError, ValueError):
                self._takes_training = True
        if self._takes_training:
            try:
                return self.obj(x, *a, training=training, **kw)
            except TypeError:
                self._takes_training = False
        return self.obj(x, *a, **kw)


class _SavedFn:
    """Callable view of a loaded SavedModel that keeps the loaded object (and its variables) alive."""

    def __init__(self, loaded, fn, kwarg=None):
        self.loaded, self.fn, self.kwarg = loaded, fn, kwarg

    @property
    def variables(self):
        return list(getattr(self.loaded, "variables", []))

    def __call__(self, x, training=False):
        if self.kwarg is None:
            return self.fn(x)
        out = self.fn(**{self.kwarg: x})
        return next(iter(out.values())) if isinstance(out, dict) and len(out) == 1 else out


def from_saved_model(path, trainable=False, signature=None, fn="serve"):
    """Load a SavedModel as a ApexNet layer (e.g. a TF-Hub style feature extractor).
    Uses the exported ``serve`` function (ApexNet exports), else the named serving ``signature``."""
    m = tf.saved_model.load(path)
    if signature is None and hasattr(m, fn):
        return Wrap(_SavedFn(m, getattr(m, fn)), trainable=trainable)
    sig = m.signatures[signature or "serving_default"]
    kwarg = next(iter(sig.structured_input_signature[1]))
    return Wrap(_SavedFn(m, sig, kwarg), trainable=trainable)


def to_keras(module, input_shape, dtype="float32"):
    """Expose a ApexNet Module as a functional Keras model (for tools that need one)."""
    inp = tf.keras.Input(shape=input_shape, dtype=dtype)
    return tf.keras.Model(inp, tf.keras.layers.Lambda(lambda t: module(t, training=False))(inp))
