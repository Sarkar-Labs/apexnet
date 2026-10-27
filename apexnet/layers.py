"""Core layers: dense, norms, dropout family, embeddings, attention, transformer, structural, remat.
Every layer is lazy-shaped, precision-policy aware and a plain tf.Module."""
import math
import tensorflow as tf
from .core import Module, Sequential, compute_dtype


def _mish(x):
    return x * tf.tanh(tf.nn.softplus(x))


ACTIVATIONS = {
    None: lambda x: x, "linear": lambda x: x, "relu": tf.nn.relu, "relu6": tf.nn.relu6,
    "gelu": lambda x: tf.nn.gelu(x, approximate=True), "gelu_exact": lambda x: tf.nn.gelu(x, approximate=False),
    "silu": tf.nn.silu, "swish": tf.nn.silu, "mish": _mish, "elu": tf.nn.elu, "selu": tf.nn.selu,
    "softplus": tf.nn.softplus, "softsign": tf.nn.softsign, "tanh": tf.nn.tanh, "sigmoid": tf.nn.sigmoid,
    "hard_sigmoid": lambda x: tf.clip_by_value(0.2 * x + 0.5, 0.0, 1.0),
    "hard_swish": lambda x: x * tf.nn.relu6(x + 3.0) / 6.0,
    "softmax": tf.nn.softmax, "log_softmax": tf.nn.log_softmax, "leaky_relu": tf.nn.leaky_relu,
}


def get_activation(a):
    return a if callable(a) else ACTIVATIONS[a]


class Activation(Module):
    def __init__(self, fn, name=None):
        super().__init__(name)
        self.fn = get_activation(fn)

    def forward(self, x, training=False):
        return self.fn(x)


class Lambda(Module):
    def __init__(self, fn, name=None):
        super().__init__(name)
        self.fn = fn

    def forward(self, x, training=False):
        return self.fn(x)


class PReLU(Module):
    def build(self, x):
        self.alpha = self.param("alpha", (x.shape[-1],), "zeros")

    def forward(self, x, training=False):
        x = self.cast(x)
        return tf.nn.relu(x) - self.cast(self.alpha) * tf.nn.relu(-x)


class GLU(Module):
    """Gated linear unit: splits last dim in two, a * sigmoid(b)."""
    def forward(self, x, training=False):
        a, b = tf.split(x, 2, axis=-1)
        return a * tf.sigmoid(b)


# ------------------------------------------------------------------ dense
class Dense(Module):
    def __init__(self, units, activation=None, use_bias=True, init="glorot_uniform",
                 regularizer=None, name=None):
        super().__init__(name)
        self.units, self.act, self.use_bias = units, get_activation(activation), use_bias
        self.init_, self.reg = init, regularizer

    def build(self, x):
        self.w = self.param("w", (x.shape[-1], self.units), self.init_, regularizer=self.reg)
        self.b = self.param("b", (self.units,), "zeros") if self.use_bias else None

    def forward(self, x, training=False, kernel=None):
        w = self.w if kernel is None else kernel
        # tensordot -> one 2-D matmul. (tf.einsum with an ellipsis is ~100x slower on CPU, fwd+bwd.)
        y = tf.tensordot(self.cast(x), self.cast(w), 1)
        if self.b is not None:
            y += self.cast(self.b)
        return self.act(y)


class Bilinear(Module):
    """y_k = x1^T W_k x2 + b_k"""
    def __init__(self, units, name=None):
        super().__init__(name)
        self.units = units

    def build(self, x):
        pass

    def forward(self, x, y=None, training=False):
        if not hasattr(self, "w"):
            self.w = self.param("w", (x.shape[-1], self.units, y.shape[-1]), 0.02)
            self.b = self.param("b", (self.units,), "zeros")
        xw = tf.tensordot(self.cast(x), self.cast(self.w), [[x.shape.rank - 1], [0]])  # [..., u, j]
        return tf.reduce_sum(xw * self.cast(y)[..., None, :], -1) + self.cast(self.b)


class Einsum(Module):
    """Learned einsum: Einsum('btd,dhk->bthk', (d, h, k))"""
    def __init__(self, equation, weight_shape, init="glorot_uniform", name=None):
        super().__init__(name)
        self.eq, self.shape, self.init_ = equation, weight_shape, init

    def build(self, x):
        self.w = self.param("w", self.shape, self.init_)

    def forward(self, x, training=False):
        return tf.einsum(self.eq, self.cast(x), self.cast(self.w))


# ------------------------------------------------------------ dropout family
class Dropout(Module):
    def __init__(self, rate, name=None):
        super().__init__(name)
        self.rate = rate

    def forward(self, x, training=False):
        return tf.nn.dropout(x, self.rate) if (training and self.rate > 0) else x


class SpatialDropout(Module):
    """Drops whole channels (works for NHWC / NLC / NDHWC)."""
    def __init__(self, rate, name=None):
        super().__init__(name)
        self.rate = rate

    def forward(self, x, training=False):
        if not training or self.rate == 0:
            return x
        shape = tf.concat([tf.shape(x)[:1], tf.ones([x.shape.rank - 2], tf.int32), tf.shape(x)[-1:]], 0)
        return tf.nn.dropout(x, self.rate, noise_shape=shape)


class GaussianNoise(Module):
    def __init__(self, stddev, name=None):
        super().__init__(name)
        self.std = stddev

    def forward(self, x, training=False):
        return x + tf.random.normal(tf.shape(x), 0.0, self.std, x.dtype) if training else x


class DropPath(Module):
    """Stochastic depth: drops the whole residual branch per sample."""
    def __init__(self, rate, name=None):
        super().__init__(name)
        self.rate = rate

    def forward(self, x, training=False):
        if not training or self.rate == 0:
            return x
        keep = 1.0 - self.rate
        shape = tf.concat([tf.shape(x)[:1], tf.ones([x.shape.rank - 1], tf.int32)], 0)
        mask = tf.floor(keep + tf.random.uniform(shape, dtype=x.dtype))
        return x / keep * mask


# -------------------------------------------------------------- normalisation
class LayerNorm(Module):
    def __init__(self, eps=1e-5, name=None):
        super().__init__(name)
        self.eps = eps

    def build(self, x):
        self.gamma = self.param("gamma", (x.shape[-1],), "ones")
        self.beta = self.param("beta", (x.shape[-1],), "zeros")

    def forward(self, x, training=False):
        xf = tf.cast(x, tf.float32)  # normalisation always in fp32
        mean, var = tf.nn.moments(xf, [-1], keepdims=True)
        return self.cast((xf - mean) * tf.math.rsqrt(var + self.eps) * self.gamma + self.beta)


class RMSNorm(Module):
    def __init__(self, eps=1e-6, name=None):
        super().__init__(name)
        self.eps = eps

    def build(self, x):
        self.gamma = self.param("gamma", (x.shape[-1],), "ones")

    def forward(self, x, training=False):
        xf = tf.cast(x, tf.float32)
        return self.cast(xf * tf.math.rsqrt(tf.reduce_mean(tf.square(xf), -1, keepdims=True) + self.eps) * self.gamma)


class BatchNorm(Module):
    def __init__(self, momentum=0.99, eps=1e-3, name=None):
        super().__init__(name)
        self.momentum, self.eps = momentum, eps

    def build(self, x):
        c = x.shape[-1]
        self.gamma = self.param("gamma", (c,), "ones")
        self.beta = self.param("beta", (c,), "zeros")
        self.mean = self.param("moving_mean", (c,), "zeros", trainable=False)
        self.var = self.param("moving_var", (c,), "ones", trainable=False)

    def forward(self, x, training=False):
        xf = tf.cast(x, tf.float32)
        if training and not self._frozen:
            axes = list(range(xf.shape.rank - 1))
            m, v = tf.nn.moments(xf, axes)
            self.mean.assign(self.momentum * self.mean + (1 - self.momentum) * m)
            self.var.assign(self.momentum * self.var + (1 - self.momentum) * v)
        else:
            m, v = self.mean, self.var
        return self.cast(tf.nn.batch_normalization(xf, m, v, self.beta, self.gamma, self.eps))


class GroupNorm(Module):
    """Group normalisation over channel groups (channels-last, any rank >= 3)."""
    def __init__(self, groups=32, eps=1e-5, name=None):
        super().__init__(name)
        self.groups, self.eps = groups, eps

    def build(self, x):
        c = x.shape[-1]
        assert c % self.groups == 0, "channels must be divisible by groups"
        self.gamma = self.param("gamma", (c,), "ones")
        self.beta = self.param("beta", (c,), "zeros")

    def forward(self, x, training=False):
        xf = tf.cast(x, tf.float32)
        shp = tf.shape(xf)
        C = x.shape[-1]
        g = tf.reshape(xf, tf.concat([shp[:-1], [self.groups, C // self.groups]], 0))
        axes = list(range(1, g.shape.rank - 2)) + [g.shape.rank - 1]
        mean, var = tf.nn.moments(g, axes, keepdims=True)
        g = (g - mean) * tf.math.rsqrt(var + self.eps)
        return self.cast(tf.reshape(g, shp) * self.gamma + self.beta)


def InstanceNorm(eps=1e-5, name=None):
    """GroupNorm with one channel per group is set at build; implemented as a thin subclass."""
    class _IN(GroupNorm):
        def build(self, x):
            self.groups = x.shape[-1]
            super().build(x)
    return _IN(1, eps, name)


class SpectralNorm(Module):
    """Wraps a Dense / Conv layer (anything with `.w` and `forward(..., kernel=)`);
    divides its kernel by an estimate of its top singular value."""
    def __init__(self, inner, iters=1, name=None):
        super().__init__(name)
        self.inner, self.iters = inner, iters

    def build(self, x):
        self.inner(x)  # build wrapped layer
        w = self.inner.w
        self.u = self.param("u", (1, int(w.shape[-1])), 1.0, trainable=False)

    def forward(self, x, training=False):
        w = self.inner.w
        w2 = tf.reshape(w, [-1, w.shape[-1]])
        u = self.u
        for _ in range(self.iters):
            v = tf.math.l2_normalize(tf.matmul(u, w2, transpose_b=True))
            u = tf.math.l2_normalize(tf.matmul(v, w2))
        sigma = tf.matmul(tf.matmul(v, w2), u, transpose_b=True)[0, 0]
        if training:
            self.u.assign(tf.stop_gradient(u))
        return self.inner.forward(x, training=training, kernel=w / sigma)


# ----------------------------------------------------------------- embedding
class Embedding(Module):
    def __init__(self, vocab, dim, init=0.02, name=None):
        super().__init__(name)
        self.vocab, self.dim, self.init_ = vocab, dim, init

    def build(self, x):
        self.table = self.param("table", (self.vocab, self.dim), self.init_)

    def forward(self, ids, training=False):
        return self.cast(tf.gather(self.table, ids))


class PositionalEncoding(Module):
    """Fixed sinusoidal positional encoding added to x [B, T, D]."""
    def forward(self, x, training=False):
        T, D = tf.shape(x)[1], x.shape[-1]
        pos = tf.cast(tf.range(T), tf.float32)[:, None]
        i = tf.cast(tf.range(D // 2), tf.float32)[None, :]
        ang = pos / tf.pow(10000.0, 2.0 * i / D)
        pe = tf.concat([tf.sin(ang), tf.cos(ang)], -1)
        return x + self.cast(pe[None])


def apply_rope(x, base=10000.0):
    """Rotary position embedding on [B, T, H, Dh] (Dh even)."""
    Dh = x.shape[-1]
    T = tf.shape(x)[1]
    inv = 1.0 / tf.pow(base, tf.cast(tf.range(0, Dh, 2), tf.float32) / Dh)
    ang = tf.cast(tf.range(T), tf.float32)[:, None] * inv[None, :]
    cos, sin = tf.cos(ang)[None, :, None, :], tf.sin(ang)[None, :, None, :]
    xf = tf.cast(x, tf.float32)
    x1, x2 = xf[..., : Dh // 2], xf[..., Dh // 2:]
    return tf.cast(tf.concat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1), x.dtype)


# ----------------------------------------------------------------- attention
class MultiHeadAttention(Module):
    """Self- or cross-attention.
    ``cross=True``: queries from x, keys/values from ``context`` [B, S, Dc].
    ``mask``: [B, T] (self) or [B, S] (cross) bool, True = real token.  ``rotary``: RoPE (self-attn)."""

    def __init__(self, heads, causal=False, dropout=0.0, cross=False, rotary=False, name=None):
        super().__init__(name)
        self.heads, self.causal, self.dropout, self.cross, self.rotary = heads, causal, dropout, cross, rotary

    def build(self, x):
        d = x.shape[-1]
        assert d % self.heads == 0, "dim must be divisible by heads"
        if self.cross:
            self.q, self.kv = Dense(d, use_bias=False), Dense(2 * d, use_bias=False)
        else:
            self.qkv = Dense(3 * d, use_bias=False)
        self.out = Dense(d)

    def forward(self, x, context=None, mask=None, training=False):
        D = x.shape[-1]
        H, Dh = self.heads, D // self.heads
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        if self.cross:
            q = self.q(x)
            k, v = tf.split(self.kv(context), 2, -1)
            S = tf.shape(context)[1]
        else:
            q, k, v = tf.split(self.qkv(x), 3, -1)
            S = T
        q = tf.reshape(q, [B, T, H, Dh]); k = tf.reshape(k, [B, S, H, Dh]); v = tf.reshape(v, [B, S, H, Dh])
        if self.rotary and not self.cross:
            q, k = apply_rope(q), apply_rope(k)
        q, k, v = (tf.transpose(t, [0, 2, 1, 3]) for t in (q, k, v))  # [B, H, T|S, Dh]
        logits = tf.cast(tf.matmul(q, k, transpose_b=True), tf.float32) / math.sqrt(Dh)
        # Masking uses tf.where, NOT `logits += (1-mask) * -1e9`: on CPU the additive broadcast fused after the batched
        # matmul is ~100x slower (measured: 94 ms vs 1 ms for a tiny causal attention). Same softmax result.
        keep = None
        if self.causal:
            keep = (tf.range(T)[:, None] >= tf.range(S)[None, :])[None, None]
        if mask is not None:
            pad = tf.cast(mask, tf.bool)[:, None, None, :]
            keep = pad if keep is None else tf.logical_and(keep, pad)
        if keep is not None:
            logits = tf.where(keep, logits, -1e9)
        attn = tf.nn.softmax(logits, -1)
        if training and self.dropout > 0:
            attn = tf.nn.dropout(attn, self.dropout)
        o = tf.transpose(tf.matmul(self.cast(attn), v), [0, 2, 1, 3])  # [B, T, H, Dh]
        return self.out(tf.reshape(o, [B, T, D]), training=training)


class TransformerBlock(Module):
    """Pre-norm transformer block; ``cross=True`` adds an encoder-decoder attention sub-layer."""

    def __init__(self, heads, mlp_ratio=4, dropout=0.0, causal=False, cross=False, rotary=False,
                 drop_path=0.0, activation="gelu", name=None):
        super().__init__(name)
        self.heads, self.ratio, self.dropout, self.causal = heads, mlp_ratio, dropout, causal
        self.cross, self.rotary, self.dp, self.activation = cross, rotary, drop_path, activation

    def build(self, x):
        d = x.shape[-1]
        self.ln1, self.ln2 = LayerNorm(), LayerNorm()
        self.attn = MultiHeadAttention(self.heads, self.causal, self.dropout, rotary=self.rotary)
        if self.cross:
            self.lnc = LayerNorm()
            self.xattn = MultiHeadAttention(self.heads, False, self.dropout, cross=True)
        self.mlp = Sequential(Dense(d * self.ratio, self.activation), Dense(d))
        self.drop, self.dpath = Dropout(self.dropout), DropPath(self.dp)

    def forward(self, x, mask=None, context=None, context_mask=None, training=False):
        x = x + self.dpath(self.drop(self.attn(self.ln1(x), mask=mask, training=training), training=training), training=training)
        if self.cross:
            x = x + self.drop(self.xattn(self.lnc(x), context=context, mask=context_mask, training=training), training=training)
        return x + self.dpath(self.drop(self.mlp(self.ln2(x), training=training), training=training), training=training)


# ----------------------------------------------------------------- structural
class Reshape(Module):
    def __init__(self, shape, name=None):
        super().__init__(name)
        self.shape = list(shape)

    def forward(self, x, training=False):
        return tf.reshape(x, [tf.shape(x)[0], *self.shape])


class Permute(Module):
    def __init__(self, dims, name=None):
        super().__init__(name)
        self.dims = [0, *dims]

    def forward(self, x, training=False):
        return tf.transpose(x, self.dims)


class Flatten(Module):
    def forward(self, x, training=False):
        return tf.reshape(x, [tf.shape(x)[0], -1])


class Residual(Module):
    def __init__(self, inner, name=None):
        super().__init__(name)
        self.inner = inner

    def forward(self, x, training=False):
        return x + self.inner(x, training=training)


class Parallel(Module):
    """Runs branches on the same input and merges: 'concat' | 'add' | 'mul' | 'stack'."""
    def __init__(self, *branches, merge="concat", axis=-1, name=None):
        super().__init__(name)
        self.branches, self.merge, self.axis = list(branches), merge, axis

    def forward(self, x, training=False):
        ys = [b(x, training=training) if isinstance(b, Module) else b(x) for b in self.branches]
        return {"concat": lambda: tf.concat(ys, self.axis), "add": lambda: tf.add_n(ys),
                "mul": lambda: tf.reduce_prod(tf.stack(ys), 0), "stack": lambda: tf.stack(ys, self.axis)}[self.merge]()


class Remat(Module):
    """Gradient checkpointing: recompute inner activations in backward to save memory
    (tf.recompute_grad). Dropout inside draws fresh randomness on recompute."""
    def __init__(self, inner, name=None):
        super().__init__(name)
        self.inner = inner

    def build(self, x):
        self.inner(x)

    def forward(self, x, training=False):
        if not training:
            return self.inner(x, training=False)
        return tf.recompute_grad(lambda a: self.inner(a, training=True))(x)
