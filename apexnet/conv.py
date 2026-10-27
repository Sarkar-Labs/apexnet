"""Convolution, pooling and resampling layers (channels-last) built on tf.nn."""
import tensorflow as tf
from .core import Module, Sequential
from .layers import get_activation


def _tup(v, n):
    return (v,) * n if isinstance(v, int) else tuple(v)


class _ConvND(Module):
    rank = 2

    def __init__(self, filters, kernel=3, stride=1, padding="same", dilation=1, activation=None,
                 use_bias=True, init="he_normal", regularizer=None, name=None):
        super().__init__(name)
        n = self.rank
        self.filters, self.kernel, self.stride, self.dilation = filters, _tup(kernel, n), _tup(stride, n), _tup(dilation, n)
        self.padding = padding.upper()
        assert self.padding in ("SAME", "VALID", "CAUSAL")
        assert self.padding != "CAUSAL" or n == 1, "causal padding is 1-D only"
        self.act, self.use_bias, self.init_, self.reg = get_activation(activation), use_bias, init, regularizer

    def build(self, x):
        self.w = self.param("w", (*self.kernel, x.shape[-1], self.filters), self.init_, regularizer=self.reg)
        self.b = self.param("b", (self.filters,), "zeros") if self.use_bias else None

    def forward(self, x, training=False, kernel=None):
        x, w = self.cast(x), self.cast(self.w if kernel is None else kernel)
        pad = self.padding
        if pad == "CAUSAL":
            x = tf.pad(x, [[0, 0], [(self.kernel[0] - 1) * self.dilation[0], 0], [0, 0]])
            pad = "VALID"
        if self.rank == 3:  # explicit op: its gradient needs data_format set (tf.nn.convolution omits it)
            y = tf.nn.conv3d(x, w, [1, *self.stride, 1], pad, data_format="NDHWC",
                             dilations=[1, *self.dilation, 1])
        else:
            y = tf.nn.convolution(x, w, strides=self.stride, padding=pad, dilations=self.dilation)
        if self.b is not None:
            y += self.cast(self.b)
        return self.act(y)


class Conv1D(_ConvND):
    rank = 1


class Conv2D(_ConvND):
    rank = 2


class Conv3D(_ConvND):
    rank = 3


class _ConvTransposeND(Module):
    rank = 2

    def __init__(self, filters, kernel=2, stride=2, padding="same", activation=None, use_bias=True,
                 init="he_normal", name=None):
        super().__init__(name)
        n = self.rank
        self.filters, self.kernel, self.stride = filters, _tup(kernel, n), _tup(stride, n)
        self.padding, self.act, self.use_bias, self.init_ = padding.upper(), get_activation(activation), use_bias, init

    def build(self, x):
        # tf.nn.conv_transpose filter layout: [*k, out_channels, in_channels]
        self.w = self.param("w", (*self.kernel, self.filters, x.shape[-1]), self.init_)
        self.b = self.param("b", (self.filters,), "zeros") if self.use_bias else None

    def forward(self, x, training=False):
        x = self.cast(x)
        sp = [tf.shape(x)[1 + i] for i in range(self.rank)]
        out = []
        for i in range(self.rank):
            s, k = self.stride[i], self.kernel[i]
            out.append(sp[i] * s if self.padding == "SAME" else sp[i] * s + max(k - s, 0))
        out_shape = tf.stack([tf.shape(x)[0], *out, self.filters])
        y = tf.nn.conv_transpose(x, self.cast(self.w), out_shape, self.stride, self.padding)
        if self.b is not None:
            y += self.cast(self.b)
        return self.act(y)


class Conv1DTranspose(_ConvTransposeND):
    rank = 1


class Conv2DTranspose(_ConvTransposeND):
    rank = 2


class Conv3DTranspose(_ConvTransposeND):
    rank = 3


class DepthwiseConv2D(Module):
    def __init__(self, kernel=3, stride=1, padding="same", multiplier=1, dilation=1, activation=None,
                 use_bias=True, name=None):
        super().__init__(name)
        self.kernel, self.stride, self.padding = _tup(kernel, 2), _tup(stride, 2), padding.upper()
        self.mult, self.dilation, self.act, self.use_bias = multiplier, _tup(dilation, 2), get_activation(activation), use_bias

    def build(self, x):
        c = x.shape[-1]
        self.w = self.param("w", (*self.kernel, c, self.mult), "he_normal")
        self.b = self.param("b", (c * self.mult,), "zeros") if self.use_bias else None

    def forward(self, x, training=False, kernel=None):
        w = self.w if kernel is None else kernel
        y = tf.nn.depthwise_conv2d(self.cast(x), self.cast(w), [1, *self.stride, 1], self.padding,
                                   dilations=self.dilation)
        if self.b is not None:
            y += self.cast(self.b)
        return self.act(y)


class SeparableConv2D(Module):
    """Depthwise 3x3 followed by pointwise 1x1 (MobileNet-style)."""
    def __init__(self, filters, kernel=3, stride=1, padding="same", activation=None, use_bias=True, name=None):
        super().__init__(name)
        self.dw = DepthwiseConv2D(kernel, stride, padding, use_bias=False)
        self.pw = Conv2D(filters, 1, activation=activation, use_bias=use_bias)

    def forward(self, x, training=False):
        return self.pw(self.dw(x))


# ------------------------------------------------------------------ pooling
class _Pool(Module):
    kind, rank = "MAX", 2

    def __init__(self, size=2, stride=None, padding="valid", name=None):
        super().__init__(name)
        self.size, self.padding = _tup(size, self.rank), padding.upper()
        self.stride = _tup(stride if stride is not None else size, self.rank)

    def forward(self, x, training=False):
        return tf.nn.pool(x, self.size, self.kind, strides=self.stride, padding=self.padding)


def _mk(kind, rank):
    return type(f"{'Max' if kind == 'MAX' else 'Avg'}Pool{rank}D", (_Pool,), dict(kind=kind, rank=rank))


MaxPool1D, MaxPool2D, MaxPool3D = (_mk("MAX", r) for r in (1, 2, 3))
AvgPool1D, AvgPool2D, AvgPool3D = (_mk("AVG", r) for r in (1, 2, 3))


class GlobalAvgPool(Module):
    """Mean over all spatial/time axes (everything between batch and channel)."""
    def forward(self, x, training=False, mask=None):
        axes = list(range(1, x.shape.rank - 1))
        if mask is None:
            return tf.reduce_mean(x, axes)
        m = tf.cast(mask, x.dtype)[..., None]
        return tf.reduce_sum(x * m, axes) / tf.maximum(tf.reduce_sum(m, axes), 1.0)


class GlobalMaxPool(Module):
    def forward(self, x, training=False):
        return tf.reduce_max(x, list(range(1, x.shape.rank - 1)))


GlobalAvgPool2D = GlobalAvgPool
GlobalAvgPool1D = GlobalAvgPool


class Upsample(Module):
    """Resize spatial dims of NHWC by an integer factor ('nearest' | 'bilinear' | 'bicubic' | 'lanczos3')."""
    def __init__(self, scale=2, method="nearest", name=None):
        super().__init__(name)
        self.scale, self.method = scale, method

    def forward(self, x, training=False):
        h, w = tf.shape(x)[1] * self.scale, tf.shape(x)[2] * self.scale
        return tf.cast(tf.image.resize(x, [h, w], method=self.method), x.dtype)


class ZeroPad(Module):
    """Pad spatial dims: ZeroPad(((top, bottom), (left, right))) for NHWC, or ZeroPad(((l, r),)) for NLC."""
    def __init__(self, pads, name=None):
        super().__init__(name)
        self.pads = [[0, 0], *[list(p) for p in pads], [0, 0]]

    def forward(self, x, training=False):
        return tf.pad(x, self.pads)


class PixelShuffle(Module):
    """Sub-pixel upsampling (depth_to_space)."""
    def __init__(self, scale=2, name=None):
        super().__init__(name)
        self.scale = scale

    def forward(self, x, training=False):
        return tf.nn.depth_to_space(x, self.scale)


class SqueezeExcite(Module):
    def __init__(self, ratio=4, name=None):
        super().__init__(name)
        self.ratio = ratio

    def build(self, x):
        from .layers import Dense
        c = x.shape[-1]
        self.fc1, self.fc2 = Dense(max(1, c // self.ratio), "relu"), Dense(c, "sigmoid")

    def forward(self, x, training=False):
        s = tf.reduce_mean(x, list(range(1, x.shape.rank - 1)), keepdims=True)
        return x * self.cast(self.fc2(self.fc1(s)))
