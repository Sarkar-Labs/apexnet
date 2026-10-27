"""Weight regularizers: pass to ``Module.param(..., regularizer=)`` or layer ``regularizer=`` args.
They are serialisable (part of ``get_config``)."""
import tensorflow as tf
from .core import Configurable, register_object


def l1(c=1e-4):
    return Configurable("l1", lambda w: c * tf.reduce_sum(tf.abs(w)), c=c)


def l2(c=1e-4):
    return Configurable("l2", lambda w: c * tf.reduce_sum(tf.square(w)), c=c)


def l1_l2(l1_c=1e-5, l2_c=1e-4):
    return Configurable("l1_l2", lambda w: l1_c * tf.reduce_sum(tf.abs(w)) + l2_c * tf.reduce_sum(tf.square(w)),
                        l1_c=l1_c, l2_c=l2_c)


def orthogonality(c=1e-3):
    """Soft orthogonality: ||W^T W - I||^2 (for 2-D weights)."""
    def fn(w):
        w2 = tf.reshape(w, [-1, w.shape[-1]])
        g = tf.matmul(w2, w2, transpose_a=True)
        return c * tf.reduce_sum(tf.square(g - tf.eye(w.shape[-1])))
    return Configurable("orthogonality", fn, c=c)


for _k, _f in (("l1", l1), ("l2", l2), ("l1_l2", l1_l2), ("orthogonality", orthogonality)):
    register_object(_k, _f)
