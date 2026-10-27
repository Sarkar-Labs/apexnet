"""Losses and metrics.

Loss contract: ``loss(y_true, y_pred)`` returns per-element losses, or a tuple
``(losses, weights)`` (e.g. to mask padding). The Trainer normalises by the
*global* weight sum, so masked / multi-device training is exactly correct.

Metric contract: ``metric(y, pred, w)`` returns ``(sum, count)`` for the batch;
the Trainer aggregates these across steps and replicas, then calls ``result``.
"""
import tensorflow as tf


def _bcast(w, like):
    if w is None:
        return None
    w = tf.cast(w, tf.float32)
    while w.shape.rank < like.shape.rank:
        w = tf.expand_dims(w, -1)
    return w


def cross_entropy(label_smoothing=0.0, ignore_index=None):
    """Sparse (integer-label) softmax cross-entropy from logits. Works for [B,C] and [B,T,V]."""
    def fn(y, logits):
        y = tf.cast(y, tf.int32)
        logits = tf.cast(logits, tf.float32)
        mask = tf.ones_like(y, tf.float32) if ignore_index is None \
            else tf.cast(tf.not_equal(y, ignore_index), tf.float32)
        safe = tf.where(mask > 0, y, tf.zeros_like(y))
        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(safe, logits)
        if label_smoothing:
            smooth = -tf.reduce_mean(tf.nn.log_softmax(logits), -1)
            nll = (1 - label_smoothing) * nll + label_smoothing * smooth
        return nll, mask
    return fn


def binary_cross_entropy(y, logits):
    y = tf.cast(y, tf.float32)
    logits = tf.cast(logits, tf.float32)
    if logits.shape.rank > y.shape.rank:
        y = y[..., None]
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, logits), -1)


def _align(y, p):
    """Cast and shape-align targets/predictions so the last axis is the per-example feature axis."""
    y, p = tf.cast(y, tf.float32), tf.cast(p, tf.float32)
    if y.shape.rank < p.shape.rank:
        y = y[..., None]
    elif p.shape.rank == 1:  # [B] vs [B]: keep the batch axis
        y, p = y[:, None], p[:, None]
    return y, p


def mse(y, p):
    y, p = _align(y, p)
    return tf.reduce_mean(tf.square(y - p), -1)


def mae(y, p):
    y, p = _align(y, p)
    return tf.reduce_mean(tf.abs(y - p), -1)


def huber(delta=1.0):
    def fn(y, p):
        y, p = _align(y, p)
        e = tf.abs(y - p)
        q = tf.minimum(e, delta)
        return tf.reduce_mean(0.5 * q * q + delta * (e - q), -1)
    return fn



def categorical_cross_entropy(label_smoothing=0.0, from_logits=True):
    """Dense / soft-label cross-entropy (one-hot, mixup, distillation targets). y: [..., C] probabilities."""
    def fn(y, logits):
        y, logits = tf.cast(y, tf.float32), tf.cast(logits, tf.float32)
        if label_smoothing:
            y = y * (1 - label_smoothing) + label_smoothing / tf.cast(tf.shape(y)[-1], tf.float32)
        logp = tf.nn.log_softmax(logits) if from_logits else tf.math.log(tf.clip_by_value(logits, 1e-9, 1.0))
        return -tf.reduce_sum(y * logp, -1)
    return fn


def focal_loss(gamma=2.0, alpha=None, ignore_index=None):
    """Sparse multi-class focal loss (Lin et al.)."""
    def fn(y, logits):
        y = tf.cast(y, tf.int32)
        logits = tf.cast(logits, tf.float32)
        mask = tf.ones_like(y, tf.float32) if ignore_index is None else tf.cast(tf.not_equal(y, ignore_index), tf.float32)
        safe = tf.where(mask > 0, y, tf.zeros_like(y))
        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(safe, logits)
        pt = tf.exp(-nll)
        loss = (1 - pt) ** gamma * nll
        if alpha is not None:
            loss *= tf.gather(tf.constant(alpha, tf.float32), safe)
        return loss, mask
    return fn


def hinge(y, p):
    """y in {0,1} or {-1,1}; p raw scores."""
    y, p = _align(y, p)
    y = tf.where(y <= 0, -tf.ones_like(y), tf.ones_like(y))
    return tf.reduce_mean(tf.nn.relu(1.0 - y * p), -1)


def squared_hinge(y, p):
    y, p = _align(y, p)
    y = tf.where(y <= 0, -tf.ones_like(y), tf.ones_like(y))
    return tf.reduce_mean(tf.square(tf.nn.relu(1.0 - y * p)), -1)


def kl_divergence(y, logits):
    """KL(y || softmax(logits)); y is a probability distribution [..., C]."""
    y = tf.clip_by_value(tf.cast(y, tf.float32), 1e-9, 1.0)
    return tf.reduce_sum(y * (tf.math.log(y) - tf.nn.log_softmax(tf.cast(logits, tf.float32))), -1)


def distillation(temperature=2.0, alpha=0.5, hard=None):
    """Knowledge distillation: y = (teacher_logits, hard_labels) is NOT used; pass teacher logits as y
    and (optionally) supply hard labels via a custom step. Plain soft-target form here."""
    def fn(teacher_logits, student_logits):
        t = temperature
        pt = tf.nn.softmax(tf.cast(teacher_logits, tf.float32) / t)
        return t * t * kl_divergence(pt, tf.cast(student_logits, tf.float32) / t)
    return fn


def poisson(y, p):
    y, p = _align(y, p)
    return tf.reduce_mean(tf.exp(p) - y * p, -1)  # p are log-rates


def log_cosh(y, p):
    y, p = _align(y, p)
    d = p - y
    return tf.reduce_mean(d + tf.nn.softplus(-2.0 * d) - tf.math.log(2.0), -1)


def cosine_distance(y, p):
    y, p = _align(y, p)
    return -tf.reduce_sum(tf.math.l2_normalize(y, -1) * tf.math.l2_normalize(p, -1), -1)


def dice_loss(smooth=1.0):
    """Soft Dice for segmentation: y [B,H,W] ints or [B,H,W,C] one-hot; logits [B,H,W,C]."""
    def fn(y, logits):
        p = tf.nn.softmax(tf.cast(logits, tf.float32))
        if y.shape.rank == p.shape.rank - 1:
            y = tf.one_hot(tf.cast(y, tf.int32), tf.shape(p)[-1])
        y = tf.cast(y, tf.float32)
        axes = list(range(1, p.shape.rank - 1))
        inter = tf.reduce_sum(p * y, axes)
        den = tf.reduce_sum(p, axes) + tf.reduce_sum(y, axes)
        return 1.0 - tf.reduce_mean((2 * inter + smooth) / (den + smooth), -1)
    return fn


def ctc_loss(blank=0, pad=-1):
    """CTC. y: [B, L] int labels padded with ``pad`` (must differ from ``blank``); logits: [B, T, C]."""
    def fn(y, logits):
        y = tf.cast(y, tf.int32)
        logits = tf.cast(logits, tf.float32)
        label_len = tf.reduce_sum(tf.cast(tf.not_equal(y, pad), tf.int32), -1)
        logit_len = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
        return tf.nn.ctc_loss(tf.where(y == pad, tf.zeros_like(y), y), logits, label_len, logit_len,
                              logits_time_major=False, blank_index=blank)
    return fn


def info_nce(temperature=0.1):
    """Symmetric contrastive loss. The model must return a pair (z1, z2) of [B, D] embeddings; y ignored."""
    def fn(y, pred):
        a, b = (tf.math.l2_normalize(tf.cast(t, tf.float32), -1) for t in pred)
        logits = tf.matmul(a, b, transpose_b=True) / temperature
        labels = tf.range(tf.shape(a)[0])
        l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, tf.transpose(logits))
        return 0.5 * (l1 + l2)
    return fn


def triplet_loss(margin=1.0):
    """pred = (anchor, positive, negative) embeddings."""
    def fn(y, pred):
        a, p, n = (tf.cast(t, tf.float32) for t in pred)
        return tf.nn.relu(tf.reduce_sum(tf.square(a - p), -1) - tf.reduce_sum(tf.square(a - n), -1) + margin)
    return fn


# ------------------------------------------------------------------- metrics
class Metric:
    """A metric returns a tuple of per-batch *sums* (scalars or fixed-shape vectors). The Trainer adds
    them across steps/replicas and calls ``result(*sums)``. ``shapes`` declares each sum's shape."""
    name = "metric"
    shapes = ((), ())

    def __call__(self, y, pred, w=None):
        raise NotImplementedError

    def result(self, s, c):
        return s / max(c, 1e-12)


class Accuracy(Metric):
    name = "acc"

    def __init__(self, ignore_index=None):
        self.ignore = ignore_index

    def __call__(self, y, pred, w=None):
        y = tf.cast(y, tf.int32)
        hit = tf.cast(tf.equal(tf.argmax(pred, -1, output_type=tf.int32), y), tf.float32)
        m = tf.ones_like(hit) if self.ignore is None else tf.cast(tf.not_equal(y, self.ignore), tf.float32)
        if w is not None:
            m *= _bcast(w, hit)
        return tf.reduce_sum(hit * m), tf.reduce_sum(m)


class TopKAccuracy(Accuracy):
    def __init__(self, k=5):
        super().__init__()
        self.k, self.name = k, f"top{k}_acc"

    def __call__(self, y, pred, w=None):
        hit = tf.cast(tf.math.in_top_k(tf.cast(y, tf.int32), tf.cast(pred, tf.float32), self.k), tf.float32)
        m = tf.ones_like(hit) if w is None else _bcast(w, hit)
        return tf.reduce_sum(hit * m), tf.reduce_sum(m)


class BinaryAccuracy(Metric):
    name = "acc"

    def __call__(self, y, pred, w=None):
        p = tf.cast(pred > 0, tf.float32)
        y = tf.cast(y, tf.float32)
        if p.shape.rank > y.shape.rank:
            y = y[..., None]
        hit = tf.reduce_mean(tf.cast(tf.equal(p, y), tf.float32), -1)
        m = tf.ones_like(hit) if w is None else _bcast(w, hit)
        return tf.reduce_sum(hit * m), tf.reduce_sum(m)


class _FromLoss(Metric):
    def __init__(self, fn, name):
        self.fn, self.name = fn, name

    def __call__(self, y, pred, w=None):
        out = self.fn(y, pred)
        lv, lw = out if isinstance(out, tuple) else (out, tf.ones_like(out))
        if w is not None:
            lw = lw * _bcast(w, lw)
        return tf.reduce_sum(lv * lw), tf.reduce_sum(lw)


class MAE(_FromLoss):
    def __init__(self):
        super().__init__(mae, "mae")


class MSE(_FromLoss):
    def __init__(self):
        super().__init__(mse, "mse")


class Perplexity(_FromLoss):
    def __init__(self, ignore_index=None):
        super().__init__(cross_entropy(ignore_index=ignore_index), "ppl")

    def result(self, s, c):
        import math
        return math.exp(min(s / max(c, 1e-12), 50.0))


def _binary_pred(y, pred):
    """Hard binary predictions + boolean targets, from logits (same rank) or 2+-class scores."""
    pred = tf.cast(pred, tf.float32)
    if pred.shape.rank == y.shape.rank + 1:
        hard = pred[..., 0] > 0 if pred.shape[-1] == 1 else tf.equal(tf.argmax(pred, -1), 1)
    else:
        hard = pred > 0
    return hard, tf.cast(y, tf.float32) > 0.5


def _wt(w, like):
    return tf.ones_like(like, tf.float32) if w is None else _bcast(w, like) * tf.ones_like(like, tf.float32)


class _PRF(Metric):
    shapes = ((), (), ())
    kind = "f1"

    def __call__(self, y, pred, w=None):
        h, t = _binary_pred(y, pred)
        wt = _wt(w, tf.cast(t, tf.float32))
        f = lambda c: tf.reduce_sum(tf.cast(c, tf.float32) * wt)
        return f(h & t), f(h & ~t), f(~h & t)  # tp, fp, fn

    def result(self, tp, fp, fn):
        d = {"precision": tp + fp, "recall": tp + fn, "f1": 2 * tp + fp + fn}[self.kind]
        return (tp if self.kind != "f1" else 2 * tp) / max(d, 1e-12)


class Precision(_PRF):
    name, kind = "precision", "precision"


class Recall(_PRF):
    name, kind = "recall", "recall"


class F1(_PRF):
    name, kind = "f1", "f1"


class MacroF1(Metric):
    """Per-class F1 averaged over classes (multi-class, argmax prediction)."""
    name = "macro_f1"

    def __init__(self, num_classes):
        self.n = num_classes
        self.shapes = ((num_classes,),) * 3

    def __call__(self, y, pred, w=None):
        yt = tf.one_hot(tf.cast(y, tf.int32), self.n)
        yp = tf.one_hot(tf.argmax(pred, -1, output_type=tf.int32), self.n)
        wt = _wt(w, tf.cast(y, tf.float32))[..., None]
        ax = list(range(yt.shape.rank - 1))
        f = lambda a: tf.reduce_sum(a * wt, ax)
        return f(yt * yp), f(yp * (1 - yt)), f((1 - yp) * yt)

    def result(self, tp, fp, fn):
        import numpy as np
        return float(np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)))


class AUC(Metric):
    """Binary ROC-AUC from score histograms (exact up to ``bins`` score resolution)."""
    name = "auc"

    def __init__(self, bins=512):
        self.bins = bins
        self.shapes = ((bins,), (bins,))

    def __call__(self, y, pred, w=None):
        pred = tf.cast(pred, tf.float32)
        if pred.shape.rank == y.shape.rank + 1:
            pred = pred[..., 0] if pred.shape[-1] == 1 else pred[..., 1] - pred[..., 0]
        p = tf.sigmoid(pred)
        b = tf.clip_by_value(tf.cast(p * self.bins, tf.int32), 0, self.bins - 1)
        t = tf.cast(y, tf.float32) > 0.5
        wt = _wt(w, p)
        seg = lambda m: tf.math.unsorted_segment_sum(tf.reshape(tf.cast(m, tf.float32) * wt, [-1]),
                                                     tf.reshape(b, [-1]), self.bins)
        return seg(t), seg(~t)

    def result(self, pos, neg):
        import numpy as np
        tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
        if tp[-1] == 0 or fp[-1] == 0:
            return float("nan")
        tpr, fpr = np.r_[0.0, tp / tp[-1]], np.r_[0.0, fp / fp[-1]]
        return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))


class R2(Metric):
    name = "r2"
    shapes = ((), (), (), ())

    def __call__(self, y, pred, w=None):
        y, p = _align(y, pred)
        wt = _wt(w, y)
        f = lambda a: tf.reduce_sum(a * wt)
        return f(tf.square(y - p)), f(y), f(tf.square(y)), tf.reduce_sum(wt)

    def result(self, sse, sy, sy2, n):
        sst = sy2 - sy * sy / max(n, 1e-12)
        return 1.0 - sse / max(sst, 1e-12)


class MeanOf(_FromLoss):
    """Turn any loss function into a reported metric: MeanOf(fn, 'name')."""
    def __init__(self, fn, name):
        super().__init__(fn, name)
