"""Optimizers and LR schedules, written in plain TF ops (no Keras dependency)."""
import math
import tensorflow as tf

_FIRST = tf.VariableAggregation.ONLY_FIRST_REPLICA
_MEAN = tf.VariableAggregation.MEAN


def _slot(v):
    return tf.Variable(tf.zeros(v.shape, v.dtype), trainable=False, aggregation=_MEAN)


# ----------------------------------------------------------------- schedules
def constant(lr):
    return lambda step: tf.constant(lr, tf.float32)


def warmup_cosine(peak, warmup_steps, total_steps, end_factor=0.0):
    end = peak * end_factor

    def fn(step):
        s = tf.cast(step, tf.float32)
        warm = peak * (s + 1.0) / max(1, warmup_steps)
        prog = tf.clip_by_value((s - warmup_steps) / max(1, total_steps - warmup_steps), 0.0, 1.0)
        cos = end + (peak - end) * 0.5 * (1.0 + tf.cos(math.pi * prog))
        return tf.where(s < warmup_steps, warm, cos)
    return fn


def warmup_linear(peak, warmup_steps, total_steps, end_factor=0.0):
    def fn(step):
        s = tf.cast(step, tf.float32)
        warm = peak * (s + 1.0) / max(1, warmup_steps)
        prog = tf.clip_by_value((s - warmup_steps) / max(1, total_steps - warmup_steps), 0.0, 1.0)
        return tf.where(s < warmup_steps, warm, peak * (1 - prog) + peak * end_factor * prog)
    return fn


def exponential_decay(lr, decay_steps, rate, staircase=False):
    def fn(step):
        p = tf.cast(step, tf.float32) / decay_steps
        return lr * rate ** (tf.floor(p) if staircase else p)
    return fn


def inverse_time_decay(lr, decay_steps, rate):
    return lambda step: lr / (1.0 + rate * tf.cast(step, tf.float32) / decay_steps)


def polynomial_decay(lr, decay_steps, end=0.0, power=1.0):
    def fn(step):
        p = tf.minimum(tf.cast(step, tf.float32), decay_steps) / decay_steps
        return (lr - end) * (1.0 - p) ** power + end
    return fn


def piecewise_constant(boundaries, values):
    """values has len(boundaries)+1 entries."""
    b = tf.constant(boundaries, tf.int64)
    v = tf.constant(values, tf.float32)
    return lambda step: v[tf.reduce_sum(tf.cast(tf.cast(step, tf.int64) >= b, tf.int32))]


def cosine_restarts(lr, first_steps, t_mul=2.0, m_mul=1.0, end_factor=0.0):
    """SGDR: cosine annealing with warm restarts."""
    def fn(step):
        s = tf.cast(step, tf.float32)
        if t_mul == 1.0:
            k = tf.floor(s / first_steps)
            prog = (s - k * first_steps) / first_steps
        else:
            k = tf.floor(tf.math.log(1.0 - s / first_steps * (1.0 - t_mul)) / tf.math.log(t_mul))
            start = first_steps * (1.0 - t_mul ** k) / (1.0 - t_mul)
            prog = (s - start) / (first_steps * t_mul ** k)
        peak = lr * m_mul ** k
        return peak * (end_factor + (1 - end_factor) * 0.5 * (1.0 + tf.cos(math.pi * prog)))
    return fn


def one_cycle(max_lr, total_steps, pct_start=0.3, div=25.0, final_div=1e4):
    up = max(1, int(total_steps * pct_start))
    lo, end = max_lr / div, max_lr / final_div

    def fn(step):
        s = tf.cast(step, tf.float32)
        a = lo + (max_lr - lo) * 0.5 * (1 - tf.cos(math.pi * tf.minimum(s / up, 1.0)))
        p = tf.clip_by_value((s - up) / max(1, total_steps - up), 0.0, 1.0)
        b = end + (max_lr - end) * 0.5 * (1 + tf.cos(math.pi * p))
        return tf.where(s < up, a, b)
    return fn


def with_warmup(schedule, warmup_steps):
    """Linear warmup in front of any schedule."""
    def fn(step):
        s = tf.cast(step, tf.float32)
        return tf.where(s < warmup_steps, schedule(0) * (s + 1.0) / warmup_steps, schedule(step - warmup_steps))
    return fn


# ----------------------------------------------------------------- optimizers
class Optimizer(tf.Module):
    """``lr`` may be a float or a schedule ``step -> lr``.
    Weight decay is decoupled and, by default, skips biases / norm params (rank < 2)."""

    def __init__(self, lr=1e-3, weight_decay=0.0, decay_filter=None):
        super().__init__()
        self.lr, self.wd = lr, weight_decay
        self.decay_filter = decay_filter or (lambda v: v.shape.rank >= 2)
        self._iterations = self._lr_scale = None  # created lazily, INSIDE the active strategy scope (Trainer)

    def _state(self):
        """Create the step counter / runtime LR multiplier. Must happen under the strategy scope so they are
        mirrored variables: created outside it, every replica would advance the shared counter (N x per step)."""
        if self._iterations is None:
            self._iterations = tf.Variable(0, dtype=tf.int64, trainable=False, aggregation=_FIRST)
            # runtime multiplier (e.g. ReduceLROnPlateau); checkpointed with the optimizer
            self._lr_scale = tf.Variable(1.0, trainable=False, aggregation=_FIRST)

    @property
    def iterations(self):
        self._state()
        return self._iterations

    @property
    def lr_scale(self):
        self._state()
        return self._lr_scale

    def current_lr(self):
        lr = self.lr(self.iterations) if callable(self.lr) else self.lr
        return tf.cast(lr, tf.float32) * self.lr_scale

    def build(self, variables):
        """Create slot variables (idempotent)."""

    def apply(self, grads, variables, enabled=None):
        """Apply one update. ``enabled`` (bool scalar tensor) masks *every* write, so a skipped
        step leaves weights, slots and the step counter untouched without using tf.cond
        (variable updates inside cond are illegal for tf.distribute replicas)."""
        raise NotImplementedError

    @staticmethod
    def _set(var, new, enabled):
        var.assign(new if enabled is None else tf.where(enabled, new, var))

    def _tick(self, enabled):
        self.iterations.assign_add(1 if enabled is None else tf.cast(enabled, tf.int64))

    def _decay(self, v, upd):
        return upd + self.wd * v if (self.wd and self.decay_filter(v)) else upd


class SGD(Optimizer):
    def __init__(self, lr=0.1, momentum=0.0, nesterov=False, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.momentum, self.nesterov, self.m = momentum, nesterov, None

    def build(self, variables):
        if self.momentum and self.m is None:
            self.m = [_slot(v) for v in variables]

    def apply(self, grads, variables, enabled=None):
        lr = self.current_lr()
        for i, (g, v) in enumerate(zip(grads, variables)):
            if self.momentum:
                m_new = self.momentum * self.m[i] + g
                self._set(self.m[i], m_new, enabled)
                g = g + self.momentum * m_new if self.nesterov else m_new
            self._set(v, v - lr * self._decay(v, g), enabled)
        self._tick(enabled)


class AdamW(Optimizer):
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(lr, weight_decay)
        self.b1, self.b2, self.eps, self.m, self.s = b1, b2, eps, None, None

    def build(self, variables):
        if self.m is None:
            self.m = [_slot(v) for v in variables]
            self.s = [_slot(v) for v in variables]

    def apply(self, grads, variables, enabled=None):
        lr = self.current_lr()
        t = tf.cast(self.iterations + 1, tf.float32)
        c1, c2 = 1 - self.b1 ** t, 1 - self.b2 ** t
        for i, (g, v) in enumerate(zip(grads, variables)):
            m = self.b1 * self.m[i] + (1 - self.b1) * g
            s = self.b2 * self.s[i] + (1 - self.b2) * tf.square(g)
            self._set(self.m[i], m, enabled)
            self._set(self.s[i], s, enabled)
            upd = (m / c1) / (tf.sqrt(s / c2) + self.eps)
            self._set(v, v - lr * self._decay(v, upd), enabled)
        self._tick(enabled)


class Lion(Optimizer):
    """Sign-momentum optimizer (Chen et al. 2023): one slot per weight, so less memory than Adam."""

    def __init__(self, lr=1e-4, b1=0.9, b2=0.99, weight_decay=0.1):
        super().__init__(lr, weight_decay)
        self.b1, self.b2, self.m = b1, b2, None

    def build(self, variables):
        if self.m is None:
            self.m = [_slot(v) for v in variables]

    def apply(self, grads, variables, enabled=None):
        lr = self.current_lr()
        for i, (g, v) in enumerate(zip(grads, variables)):
            upd = tf.sign(self.b1 * self.m[i] + (1 - self.b1) * g)
            self._set(self.m[i], self.b2 * self.m[i] + (1 - self.b2) * g, enabled)
            self._set(v, v - lr * self._decay(v, upd), enabled)
        self._tick(enabled)


class RMSprop(Optimizer):
    def __init__(self, lr=1e-3, rho=0.9, momentum=0.0, eps=1e-7, centered=False, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.rho, self.momentum, self.eps, self.centered = rho, momentum, eps, centered
        self.ms = self.mom = self.mg = None

    def build(self, variables):
        if self.ms is None:
            self.ms = [_slot(v) for v in variables]
            self.mom = [_slot(v) for v in variables] if self.momentum else []
            self.mg = [_slot(v) for v in variables] if self.centered else []

    def apply(self, grads, variables, enabled=None):
        lr = self.current_lr()
        for i, (g, v) in enumerate(zip(grads, variables)):
            ms = self.rho * self.ms[i] + (1 - self.rho) * tf.square(g)
            self._set(self.ms[i], ms, enabled)
            denom = ms
            if self.centered:
                mg = self.rho * self.mg[i] + (1 - self.rho) * g
                self._set(self.mg[i], mg, enabled)
                denom = ms - tf.square(mg)
            upd = g * tf.math.rsqrt(denom + self.eps)
            if self.momentum:
                mom = self.momentum * self.mom[i] + upd
                self._set(self.mom[i], mom, enabled)
                upd = mom
            self._set(v, v - lr * self._decay(v, upd), enabled)
        self._tick(enabled)


class Adagrad(Optimizer):
    def __init__(self, lr=1e-2, initial_accumulator=0.1, eps=1e-7, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.init_acc, self.eps, self.acc = initial_accumulator, eps, None

    def build(self, variables):
        if self.acc is None:
            self.acc = [tf.Variable(tf.fill(v.shape, self.init_acc), trainable=False, aggregation=_MEAN)
                        for v in variables]

    def apply(self, grads, variables, enabled=None):
        lr = self.current_lr()
        for i, (g, v) in enumerate(zip(grads, variables)):
            a = self.acc[i] + tf.square(g)
            self._set(self.acc[i], a, enabled)
            self._set(v, v - lr * self._decay(v, g * tf.math.rsqrt(a + self.eps)), enabled)
        self._tick(enabled)


class Adamax(Optimizer):
    def __init__(self, lr=2e-3, b1=0.9, b2=0.999, eps=1e-7, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.b1, self.b2, self.eps, self.m, self.u = b1, b2, eps, None, None

    def build(self, variables):
        if self.m is None:
            self.m = [_slot(v) for v in variables]
            self.u = [_slot(v) for v in variables]

    def apply(self, grads, variables, enabled=None):
        lr = self.current_lr()
        t = tf.cast(self.iterations + 1, tf.float32)
        for i, (g, v) in enumerate(zip(grads, variables)):
            m = self.b1 * self.m[i] + (1 - self.b1) * g
            u = tf.maximum(self.b2 * self.u[i], tf.abs(g))
            self._set(self.m[i], m, enabled)
            self._set(self.u[i], u, enabled)
            self._set(v, v - lr / (1 - self.b1 ** t) * self._decay(v, m / (u + self.eps)), enabled)
        self._tick(enabled)


def Adam(lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """Adam without weight decay."""
    return AdamW(lr, b1, b2, eps, weight_decay=0.0)


class Lookahead(tf.Module):
    """Lookahead (Zhang et al. 2019): wraps any optimizer; every ``k`` steps the slow weights move ``alpha`` of the
    way to the fast weights and the fast weights are reset to them. Branch-free, so it is legal under tf.distribute."""

    def __init__(self, inner, k=5, alpha=0.5):
        super().__init__()
        self.inner, self.k, self.alpha, self.slow = inner, k, alpha, None

    # the Trainer / callbacks talk to these four attributes
    iterations = property(lambda self: self.inner.iterations)
    lr_scale = property(lambda self: self.inner.lr_scale)
    lr = property(lambda self: self.inner.lr, lambda self, v: setattr(self.inner, "lr", v))

    def current_lr(self):
        return self.inner.current_lr()

    def _state(self):
        self.inner._state()

    def build(self, variables):
        self.inner._state()
        self.inner.build(variables)
        if self.slow is None:
            self.slow = [tf.Variable(v.read_value(), trainable=False, aggregation=_MEAN) for v in variables]

    def apply(self, grads, variables, enabled=None):
        self.inner.apply(grads, variables, enabled)
        step = self.inner.iterations
        sync = tf.equal(step % self.k, 0) & (step > 0)
        if enabled is not None:
            sync &= enabled
        for v, s in zip(variables, self.slow):
            new_slow = tf.where(sync, s + self.alpha * (v - s), s)
            s.assign(new_slow)
            v.assign(tf.where(sync, new_slow, v))
