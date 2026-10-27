"""Ready-made architectures, built only from apexnet layers."""
import math
import tensorflow as tf
from .conv import (AvgPool2D, Conv2D, Conv2DTranspose, GlobalAvgPool, MaxPool2D)
from .core import Module, Sequential
from .layers import (Activation, BatchNorm, Dense, Dropout, Embedding, LayerNorm, TransformerBlock)
from .rnn import Bidirectional, LSTMCell


def MLP(hidden, out, activation="gelu", dropout=0.0):
    layers = []
    for h in hidden:
        layers += [Dense(h, activation), Dropout(dropout)]
    return Sequential(*layers, Dense(out))


class TransformerLM(Module):
    """Decoder-only language model with weight-tied output head."""

    def __init__(self, vocab, dim=128, depth=4, heads=4, max_len=256, dropout=0.0, rotary=False, name=None):
        super().__init__(name)
        self.vocab, self.dim, self.max_len, self.rotary = vocab, dim, max_len, rotary
        self.tok = Embedding(vocab, dim)
        self.pos = None if rotary else self.param("pos", (max_len, dim), 0.02)
        self.drop = Dropout(dropout)
        self.blocks = [TransformerBlock(heads, 4, dropout, causal=True, rotary=rotary) for _ in range(depth)]
        self.norm = LayerNorm()

    def forward(self, ids, training=False):
        x = self.tok(ids)
        if self.pos is not None:
            x = x + self.cast(self.pos[: tf.shape(ids)[1]])
        x = self.drop(x, training=training)
        for b in self.blocks:
            x = b(x, training=training)
        return tf.einsum("btd,vd->btv", self.norm(x), self.cast(self.tok.table))


class _BasicBlock(Module):
    def __init__(self, ch, stride=1):
        super().__init__()
        self.ch, self.stride = ch, stride

    def build(self, x):
        self.c1, self.b1 = Conv2D(self.ch, 3, self.stride, use_bias=False), BatchNorm()
        self.c2, self.b2 = Conv2D(self.ch, 3, 1, use_bias=False), BatchNorm()
        self.proj = (Sequential(Conv2D(self.ch, 1, self.stride, use_bias=False), BatchNorm())
                     if self.stride != 1 or x.shape[-1] != self.ch else None)

    def forward(self, x, training=False):
        y = tf.nn.relu(self.b1(self.c1(x), training=training))
        y = self.b2(self.c2(y), training=training)
        sc = x if self.proj is None else self.proj(x, training=training)
        return tf.nn.relu(y + sc)


class ResNet(Module):
    """ResNet-style CNN. stem='cifar' (3x3) or 'imagenet' (7x7/2 + maxpool)."""

    def __init__(self, num_classes, blocks=(2, 2, 2, 2), widths=(64, 128, 256, 512), stem="cifar", name=None):
        super().__init__(name)
        if stem == "cifar":
            self.stem = Sequential(Conv2D(widths[0], 3, use_bias=False), BatchNorm(), Activation("relu"))
        else:
            self.stem = Sequential(Conv2D(widths[0], 7, 2, use_bias=False), BatchNorm(), Activation("relu"),
                                   MaxPool2D(3, 2, "same"))
        self.stages = [_BasicBlock(w, 1 if (i == 0 and j == 0) else (2 if j == 0 else 1))
                       for i, (n, w) in enumerate(zip(blocks, widths)) for j in range(n)]
        self.pool, self.head = GlobalAvgPool(), Dense(num_classes)

    def forward(self, x, training=False):
        x = self.stem(x, training=training)
        for b in self.stages:
            x = b(x, training=training)
        return self.head(self.pool(x))


class ViT(Module):
    """Vision Transformer (patchify -> [CLS] + positions -> pre-norm blocks -> head)."""

    def __init__(self, num_classes, image_size=32, patch=4, dim=64, depth=4, heads=4, dropout=0.0, name=None):
        super().__init__(name)
        self.dim = dim
        self.embed = Conv2D(dim, patch, patch, "valid")
        n = (image_size // patch) ** 2
        self.cls = self.param("cls", (1, 1, dim), 0.02)
        self.pos = self.param("pos", (n + 1, dim), 0.02)
        self.blocks = [TransformerBlock(heads, 4, dropout) for _ in range(depth)]
        self.norm, self.head = LayerNorm(), Dense(num_classes)

    def forward(self, x, training=False):
        p = self.embed(x)
        B = tf.shape(p)[0]
        x = tf.concat([tf.tile(self.cast(self.cls), [B, 1, 1]), tf.reshape(p, [B, -1, self.dim])], 1)
        x = x + self.cast(self.pos)
        for b in self.blocks:
            x = b(x, training=training)
        return self.head(self.norm(x)[:, 0])


class UNet(Module):
    """Small U-Net for dense prediction (segmentation / denoising). Input H, W must divide by 2**depth."""

    def __init__(self, out_channels=1, base=16, depth=3, name=None):
        super().__init__(name)
        blk = lambda c: Sequential(Conv2D(c, 3, activation="relu"), Conv2D(c, 3, activation="relu"))
        self.down = [blk(base * 2 ** i) for i in range(depth)]
        self.mid = blk(base * 2 ** depth)
        self.up = [Conv2DTranspose(base * 2 ** i, 2, 2) for i in reversed(range(depth))]
        self.dec = [blk(base * 2 ** i) for i in reversed(range(depth))]
        self.pool, self.out = MaxPool2D(2), Conv2D(out_channels, 1)

    def forward(self, x, training=False):
        skips = []
        for d in self.down:
            x = d(x, training=training)
            skips.append(x)
            x = self.pool(x)
        x = self.mid(x, training=training)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = dec(tf.concat([up(x), s], -1), training=training)
        return self.out(x)


class BiLSTMClassifier(Module):
    """Embedding -> BiLSTM (padding-aware, id 0 = pad) -> masked mean pool -> linear."""

    def __init__(self, vocab, num_classes, dim=64, units=64, dropout=0.1, name=None):
        super().__init__(name)
        self.emb, self.drop = Embedding(vocab, dim), Dropout(dropout)
        self.rnn = Bidirectional(LSTMCell(units), return_sequences=True)
        self.head = Dense(num_classes)

    def forward(self, ids, training=False):
        mask = tf.not_equal(ids, 0)
        h = self.rnn(self.drop(self.emb(ids), training=training), mask=mask, training=training)
        m = tf.cast(mask, h.dtype)[..., None]
        pooled = tf.reduce_sum(h * m, 1) / tf.maximum(tf.reduce_sum(m, 1), 1.0)
        return self.head(self.drop(pooled, training=training))


class Seq2SeqTransformer(Module):
    """Encoder-decoder Transformer. Call with (src_ids, tgt_in_ids); ids 0 = padding. Train with
    batches ((src, tgt_in), tgt_out) and cross_entropy(ignore_index=0)."""

    def __init__(self, src_vocab, tgt_vocab, dim=64, depth=2, heads=4, max_len=128, dropout=0.0, name=None):
        super().__init__(name)
        self.src_emb, self.tgt_emb = Embedding(src_vocab, dim), Embedding(tgt_vocab, dim)
        self.pos = self.param("pos", (max_len, dim), 0.02)
        self.enc = [TransformerBlock(heads, 4, dropout) for _ in range(depth)]
        self.dec = [TransformerBlock(heads, 4, dropout, causal=True, cross=True) for _ in range(depth)]
        self.enorm, self.dnorm, self.out = LayerNorm(), LayerNorm(), Dense(tgt_vocab)

    def encode(self, src, training=False):
        x = self.src_emb(src) + self.cast(self.pos[: tf.shape(src)[1]])
        m = tf.not_equal(src, 0)
        for b in self.enc:
            x = b(x, mask=m, training=training)
        return self.enorm(x), m

    def forward(self, inputs, training=False):
        src, tgt = inputs
        mem, m = self.encode(src, training)
        x = self.tgt_emb(tgt) + self.cast(self.pos[: tf.shape(tgt)[1]])
        for b in self.dec:
            x = b(x, context=mem, context_mask=m, training=training)
        return self.out(self.dnorm(x))
