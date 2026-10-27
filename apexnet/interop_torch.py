"""Use PyTorch modules inside ApexNet: ``an.Torch(torch_module)``.

The torch parameters are mirrored as ``tf.Variable`` s (ApexNet optimizers / EMA / checkpoints / mixed precision all apply):
before every call the TF values are copied into the torch module, torch computes forward AND backward (its autograd graph
is kept until TensorFlow asks for the gradient), and the gradients are handed back to TensorFlow. Buffers (e.g. BatchNorm
running statistics) are mirrored too and updated in training mode.

    block = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 10))
    model = an.Sequential(an.Dense(32, "relu"), an.Torch(block))

Limits (it round-trips through the host with ``tf.py_function``): CPU/host execution only, single replica, no XLA / TPU,
one output tensor, float32 compute, not JSON-serialisable (checkpoints work). Intended for reusing existing torch
blocks / research code, not for peak speed.
"""
import collections
import itertools
import threading
import numpy as np
import tensorflow as tf
from .core import Module


class _State:  # plain holder, invisible to tf.Module tracking
    def __init__(self):
        self.lock, self.cache, self.counter = threading.Lock(), collections.OrderedDict(), itertools.count()


class Torch(Module):
    MAX_CACHE = 64  # autograd graphs kept alive between forward and backward

    def __init__(self, module, name=None):
        super().__init__(name)
        try:
            import torch
        except ImportError as e:  # pragma: no cover
            raise ImportError("an.Torch needs `pip install torch`") from e
        self.t = module.float().cpu()
        self._st = _State()
        self._params = list(self.t.parameters())
        self._bufs = [b for b in self.t.buffers() if b.is_floating_point()]
        self._out_shape = None
        self.vs, self.bvs = [], []

    def build(self, x):
        import torch
        x = tf.convert_to_tensor(x)
        self.vs = [self.param(f"p{i}", tuple(p.shape), init=(lambda shape, dtype=tf.float32, a=p.detach().numpy().copy(): tf.constant(a, dtype)))
                   for i, p in enumerate(self._params)]
        self.bvs = [self.param(f"b{i}", tuple(b.shape), init=(lambda shape, dtype=tf.float32, a=b.detach().numpy().copy(): tf.constant(a, dtype)),
                               trainable=False) for i, b in enumerate(self._bufs)]
        shape = [1] + [int(d) for d in x.shape[1:]]
        probe = torch.zeros(shape, dtype=torch.float32 if x.dtype.is_floating else torch.long)
        was = self.t.training
        self.t.eval()
        with torch.no_grad():
            out = self.t(probe)
        self.t.train(was)
        if isinstance(out, (tuple, list)):
            raise TypeError("an.Torch supports a single output tensor")
        self._out_shape = [int(d) for d in out.shape[1:]]

    def forward(self, x, training=False):
        import torch
        ctx = tf.distribute.get_replica_context()
        if ctx is not None and ctx.num_replicas_in_sync > 1:
            raise RuntimeError("an.Torch layers run on the host: single replica only")
        x = tf.convert_to_tensor(x)
        float_x = x.dtype.is_floating
        if float_x:
            x = tf.cast(x, tf.float32)
        np_, nb = len(self.vs), len(self.bvs)
        st, mod, params, bufs, out_shape = self._st, self.t, self._params, self._bufs, self._out_shape
        training = bool(training)

        def fwd_py(x, *pb):
            with st.lock:
                with torch.no_grad():
                    for p, a in zip(params, pb[:np_]):
                        p.copy_(torch.from_numpy(a.numpy().copy()))
                    for b, a in zip(bufs, pb[np_:]):
                        b.copy_(torch.from_numpy(a.numpy().copy()))
                mod.train(training)
                xt = torch.from_numpy(x.numpy().copy())
                if training:
                    if float_x:
                        xt.requires_grad_(True)
                    out = mod(xt)
                    token = next(st.counter)
                    st.cache[token] = (xt, out)
                    while len(st.cache) > Torch.MAX_CACHE:
                        st.cache.popitem(last=False)
                else:
                    with torch.no_grad():
                        out = mod(xt)
                    token = -1
                new_b = [b.detach().numpy().copy() for b in bufs]
                return [out.detach().numpy().astype(np.float32), np.int64(token)] + new_b

        def bwd_py(dy, token):
            with st.lock:
                xt, out = st.cache.pop(int(token.numpy()))
                inputs = ([xt] if float_x else []) + params
                grads = torch.autograd.grad(out, inputs, grad_outputs=torch.from_numpy(dy.numpy().copy()), allow_unused=True)
                return [np.zeros(i.shape, np.float32) if g is None else g.numpy().copy() for g, i in zip(grads, inputs)]

        @tf.custom_gradient
        def call(x, *pvals):
            outs = tf.py_function(fwd_py, [x, *pvals, *[b.read_value() for b in self.bvs]],
                                  Tout=[tf.float32, tf.int64] + [tf.float32] * nb)
            y, token, new_b = outs[0], outs[1], outs[2:]
            y.set_shape([None] + out_shape)

            def grad(dy, *_unused):  # one upstream grad per output (y and the buffer outputs)
                gs = tf.py_function(bwd_py, [tf.cast(dy, tf.float32), token], Tout=[tf.float32] * ((1 if float_x else 0) + np_))
                gs = list(gs) if isinstance(gs, (list, tuple)) else [gs]
                for g, ref in zip(gs, ([x] if float_x else []) + list(pvals)):
                    g.set_shape(ref.shape)
                gx, gp = (gs[0], gs[1:]) if float_x else (None, gs)
                return (gx, *gp)
            return (y, new_b), grad

        y, new_b = call(x, *[v.read_value() for v in self.vs])
        if training:
            for b, nv in zip(self.bvs, new_b):
                b.assign(nv)
        return y
