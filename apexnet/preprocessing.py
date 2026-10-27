"""Preprocessing layers on tf.image / tf.strings / tf.lookup / tf.signal.
Random layers are active only when ``training=True``."""
import numpy as np
import tensorflow as tf
from .core import Module


class Rescaling(Module):
    def __init__(self, scale=1 / 255.0, offset=0.0, name=None):
        super().__init__(name)
        self.scale, self.offset = scale, offset

    def forward(self, x, training=False):
        return self.cast(tf.cast(x, tf.float32) * self.scale + self.offset)


class Normalization(Module):
    """Feature-wise standardisation. Call ``adapt(data)`` once (array or tf.data.Dataset of x)."""
    def __init__(self, axis=-1, eps=1e-7, name=None):
        super().__init__(name)
        self.eps, self.axis = eps, axis

    def build(self, x):
        c = x.shape[-1]
        self.mean = self.param("mean", (c,), "zeros", trainable=False)
        self.var = self.param("var", (c,), "ones", trainable=False)

    def adapt(self, data):
        if isinstance(data, tf.data.Dataset):
            data = np.concatenate([np.asarray(b[0] if isinstance(b, tuple) else b) for b in data])
        data = np.asarray(data, np.float64)
        flat = data.reshape(-1, data.shape[-1])
        if not self._built:
            self.build(tf.zeros([1, data.shape[-1]]))
            self._built = True
        self.mean.assign(flat.mean(0).astype("float32"))
        self.var.assign(flat.var(0).astype("float32"))
        return self

    def forward(self, x, training=False):
        return self.cast((tf.cast(x, tf.float32) - self.mean) * tf.math.rsqrt(self.var + self.eps))


class Resizing(Module):
    def __init__(self, height, width, method="bilinear", antialias=False, name=None):
        super().__init__(name)
        self.size, self.method, self.aa = [height, width], method, antialias

    def forward(self, x, training=False):
        return self.cast(tf.image.resize(x, self.size, method=self.method, antialias=self.aa))


class RandomFlip(Module):
    def __init__(self, mode="horizontal", name=None):
        super().__init__(name)
        self.mode = mode

    def forward(self, x, training=False):
        if not training:
            return x
        B = tf.shape(x)[0]
        if self.mode in ("horizontal", "both"):
            x = tf.where(tf.random.uniform([B, 1, 1, 1]) < 0.5, tf.reverse(x, [2]), x)
        if self.mode in ("vertical", "both"):
            x = tf.where(tf.random.uniform([B, 1, 1, 1]) < 0.5, tf.reverse(x, [1]), x)
        return x


class RandomCrop(Module):
    """Random crop (per image) in training; center crop otherwise."""
    def __init__(self, height, width, name=None):
        super().__init__(name)
        self.h, self.w = height, width

    def forward(self, x, training=False):
        if not training:
            return tf.image.resize_with_crop_or_pad(x, self.h, self.w)
        C = x.shape[-1]
        return tf.map_fn(lambda im: tf.image.random_crop(im, [self.h, self.w, C]), x)


class RandomTranslation(Module):
    """Pad-and-crop jitter by up to ``pad`` pixels (CIFAR-style)."""
    def __init__(self, pad=4, name=None):
        super().__init__(name)
        self.pad = pad

    def forward(self, x, training=False):
        if not training:
            return x
        H, W, C = x.shape[1], x.shape[2], x.shape[3]
        p = tf.pad(x, [[0, 0], [self.pad] * 2, [self.pad] * 2, [0, 0]], mode="REFLECT")
        return tf.map_fn(lambda im: tf.image.random_crop(im, [H, W, C]), p)


class RandomBrightness(Module):
    def __init__(self, delta=0.2, name=None):
        super().__init__(name)
        self.delta = delta

    def forward(self, x, training=False):
        if not training:
            return x
        return x + tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -self.delta, self.delta, x.dtype)


class RandomContrast(Module):
    def __init__(self, lower=0.8, upper=1.2, name=None):
        super().__init__(name)
        self.lo, self.hi = lower, upper

    def forward(self, x, training=False):
        if not training:
            return x
        f = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], self.lo, self.hi, x.dtype)
        m = tf.reduce_mean(x, [1, 2, 3], keepdims=True)
        return (x - m) * f + m


class RandomRotation(Module):
    """Rotation by up to ``max_degrees`` using tf.raw_ops-free affine sampling (bilinear)."""
    def __init__(self, max_degrees=15.0, name=None):
        super().__init__(name)
        self.max = max_degrees * np.pi / 180.0

    def forward(self, x, training=False):
        if not training:
            return x
        B, H, W = tf.shape(x)[0], x.shape[1], x.shape[2]
        a = tf.random.uniform([B], -self.max, self.max)
        ys, xs = tf.meshgrid(tf.linspace(-1.0, 1.0, H), tf.linspace(-1.0, 1.0, W), indexing="ij")
        cos, sin = tf.cos(a)[:, None, None], tf.sin(a)[:, None, None]
        sx, sy = cos * xs - sin * ys, sin * xs + cos * ys
        gx, gy = (sx + 1) * 0.5 * (W - 1), (sy + 1) * 0.5 * (H - 1)
        return _bilinear(x, gx, gy)


def _bilinear(x, gx, gy):
    """Sample x [B,H,W,C] at float pixel coords gx, gy [B,H,W] (zero outside)."""
    H, W = tf.shape(x)[1], tf.shape(x)[2]
    x0, y0 = tf.floor(gx), tf.floor(gy)
    wx, wy = (gx - x0)[..., None], (gy - y0)[..., None]

    def px(xi, yi):
        inside = tf.cast((xi >= 0) & (xi < tf.cast(W, tf.float32)) & (yi >= 0) & (yi < tf.cast(H, tf.float32)), x.dtype)
        v = tf.gather_nd(x, tf.stack([tf.cast(tf.clip_by_value(yi, 0, tf.cast(H - 1, tf.float32)), tf.int32),
                                      tf.cast(tf.clip_by_value(xi, 0, tf.cast(W - 1, tf.float32)), tf.int32)], -1),
                         batch_dims=1)
        return v * inside[..., None]
    wx, wy = tf.cast(wx, x.dtype), tf.cast(wy, x.dtype)
    return (px(x0, y0) * (1 - wx) * (1 - wy) + px(x0 + 1, y0) * wx * (1 - wy)
            + px(x0, y0 + 1) * (1 - wx) * wy + px(x0 + 1, y0 + 1) * wx * wy)


class TextVectorizer(Module):
    """strings -> int ids [B, max_len]. 0 = padding, 1 = out-of-vocabulary.
    Built from tf.strings + tf.lookup, so it exports inside a SavedModel."""

    def __init__(self, max_len=64, vocab=None, lower=True, strip_punct=True, name=None):
        super().__init__(name)
        self.max_len, self.lower, self.strip = max_len, lower, strip_punct
        self.table = None
        self.vocab = []
        if vocab is not None:
            self.set_vocab(vocab)

    def _standardize(self, s):
        if self.lower:
            s = tf.strings.lower(s)
        if self.strip:
            s = tf.strings.regex_replace(s, r"[^\w\s]", "")
        return s

    def set_vocab(self, words):
        self.vocab = ["", "[UNK]"] + list(words)
        keys = tf.constant(self.vocab[2:])
        vals = tf.range(2, 2 + len(words), dtype=tf.int64)
        self.table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, vals), 1)
        return self

    def adapt(self, texts, max_tokens=20000):
        from collections import Counter
        c = Counter()
        for t in texts:
            t = t.decode() if isinstance(t, bytes) else str(t)
            c.update(self._standardize(tf.constant(t)).numpy().decode().split())
        return self.set_vocab([w for w, _ in c.most_common(max_tokens - 2)])

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_state(self):
        return {"vocab": self.vocab[2:]} if self.table is not None else None

    def set_state(self, state):
        self.set_vocab(state["vocab"])

    def forward(self, texts, training=False):
        assert self.table is not None, "call adapt(texts) or pass vocab="
        toks = tf.strings.split(self._standardize(tf.convert_to_tensor(texts)))
        ids = tf.ragged.map_flat_values(lambda t: self.table.lookup(t), toks)
        return tf.cast(ids.to_tensor(0, shape=[None, self.max_len]), tf.int32)


class Spectrogram(Module):
    """Waveform [B, N] -> magnitude STFT [B, frames, bins] via tf.signal."""
    def __init__(self, frame_length=256, frame_step=128, fft_length=None, power=2.0, name=None):
        super().__init__(name)
        self.fl, self.fs, self.fft, self.power = frame_length, frame_step, fft_length, power

    def forward(self, x, training=False):
        s = tf.signal.stft(tf.cast(x, tf.float32), self.fl, self.fs, self.fft)
        return tf.abs(s) ** self.power


class MelSpectrogram(Module):
    """Waveform -> (log-)mel spectrogram [B, frames, n_mels]."""
    def __init__(self, sample_rate=16000, n_mels=64, frame_length=400, frame_step=160, fft_length=512,
                 fmin=0.0, fmax=None, log=True, name=None):
        super().__init__(name)
        self.spec = Spectrogram(frame_length, frame_step, fft_length, 2.0)
        self.mel = tf.signal.linear_to_mel_weight_matrix(n_mels, fft_length // 2 + 1, sample_rate, fmin,
                                                         fmax or sample_rate / 2.0)
        self.log = log

    def forward(self, x, training=False):
        m = tf.tensordot(self.spec(x), self.mel, 1)
        return tf.math.log(m + 1e-6) if self.log else m
