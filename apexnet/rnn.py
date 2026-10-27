"""Recurrent layers: cells + a generic, masked, tf.scan-based RNN driver."""
import tensorflow as tf
from .core import Module


class SimpleRNNCell(Module):
    def __init__(self, units, activation=tf.tanh, name=None):
        super().__init__(name)
        self.units, self.act = units, activation

    def build(self, x):
        F, U = x.shape[-1], self.units
        self.kernel = self.param("kernel", (F, U), "glorot_uniform")
        self.rec = self.param("recurrent", (U, U), "orthogonal")
        self.bias = self.param("bias", (U,), "zeros")

    def initial_state(self, B):
        return tf.zeros([B, self.units], tf.float32)

    def forward(self, x, state, training=False):
        h = self.act(tf.matmul(self.cast(x), self.cast(self.kernel)) + tf.matmul(self.cast(state), self.cast(self.rec))
                     + self.cast(self.bias))
        return h, h


class LSTMCell(Module):
    def __init__(self, units, name=None):
        super().__init__(name)
        self.units = units

    def build(self, x):
        F, U = x.shape[-1], self.units
        self.kernel = self.param("kernel", (F, 4 * U), "glorot_uniform")
        self.rec = self.param("recurrent", (U, 4 * U), "orthogonal")
        # forget-gate bias initialised to 1 (standard trick for gradient flow)
        self.bias = self.param("bias", (4 * U,), lambda shape, dtype=tf.float32: tf.concat(
            [tf.zeros([U]), tf.ones([U]), tf.zeros([2 * U])], 0))

    def initial_state(self, B):
        z = tf.zeros([B, self.units], tf.float32)
        return (z, z)

    def forward(self, x, state, training=False):
        h, c = state
        z = (tf.matmul(self.cast(x), self.cast(self.kernel)) + tf.matmul(self.cast(h), self.cast(self.rec))
             + self.cast(self.bias))
        i, f, g, o = tf.split(tf.cast(z, tf.float32), 4, -1)
        c = tf.sigmoid(f) * tf.cast(c, tf.float32) + tf.sigmoid(i) * tf.tanh(g)
        h = tf.sigmoid(o) * tf.tanh(c)
        return h, (h, c)


class GRUCell(Module):
    def __init__(self, units, name=None):
        super().__init__(name)
        self.units = units

    def build(self, x):
        F, U = x.shape[-1], self.units
        self.kernel = self.param("kernel", (F, 3 * U), "glorot_uniform")
        self.rec = self.param("recurrent", (U, 3 * U), "orthogonal")
        self.bias = self.param("bias", (3 * U,), "zeros")
        self.rbias = self.param("recurrent_bias", (3 * U,), "zeros")

    def initial_state(self, B):
        return tf.zeros([B, self.units], tf.float32)

    def forward(self, x, state, training=False):
        xz = tf.cast(tf.matmul(self.cast(x), self.cast(self.kernel)) + self.cast(self.bias), tf.float32)
        hz = tf.cast(tf.matmul(self.cast(state), self.cast(self.rec)) + self.cast(self.rbias), tf.float32)
        xr, xu, xh = tf.split(xz, 3, -1)
        hr, hu, hh = tf.split(hz, 3, -1)
        r, u = tf.sigmoid(xr + hr), tf.sigmoid(xu + hu)
        n = tf.tanh(xh + r * hh)
        h = u * tf.cast(state, tf.float32) + (1 - u) * n
        return h, h


class RNN(Module):
    """Runs ``cell`` over time. x: [B, T, F]; mask: [B, T] bool (True = real step).
    Masked steps carry state and output forward, so the last output is the last *valid* one."""

    def __init__(self, cell, return_sequences=False, return_state=False, reverse=False, name=None):
        super().__init__(name)
        self.cell, self.seq, self.ret_state, self.reverse = cell, return_sequences, return_state, reverse

    def build(self, x):
        self.cell.build(x[:, 0])  # create cell variables eagerly, outside the scan body
        self.cell._built = True

    def forward(self, x, mask=None, initial_state=None, training=False):
        x = tf.convert_to_tensor(x)
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        state0 = self.cell.initial_state(B) if initial_state is None else initial_state
        xs = tf.transpose(x, [1, 0, 2])
        ms = tf.ones([T, B], tf.bool) if mask is None else tf.transpose(tf.cast(mask, tf.bool), [1, 0])
        out0 = tf.zeros([B, self.cell.units], tf.float32)

        def step(carry, inp):
            state, prev_out = carry
            xt, mt = inp
            out, new_state = self.cell(xt, state, training=training)
            out = tf.cast(out, tf.float32)
            m = mt[:, None]
            new_state = tf.nest.map_structure(lambda n, o: tf.where(m, tf.cast(n, tf.float32), o), new_state, state)
            return new_state, tf.where(m, out, prev_out)

        states, outs = tf.scan(step, (xs, ms), initializer=(state0, out0), reverse=self.reverse)
        last = 0 if self.reverse else -1
        final_state = tf.nest.map_structure(lambda s: s[last], states)
        y = tf.transpose(outs, [1, 0, 2]) if self.seq else outs[last]
        y = tf.cast(y, compute_dtype_of(x))
        return (y, final_state) if self.ret_state else y


def compute_dtype_of(x):
    from .core import compute_dtype
    return compute_dtype() if x.dtype.is_floating else tf.float32


def LSTM(units, **kw):
    return RNN(LSTMCell(units), **kw)


def GRU(units, **kw):
    return RNN(GRUCell(units), **kw)


def SimpleRNN(units, **kw):
    return RNN(SimpleRNNCell(units), **kw)


class Bidirectional(Module):
    """Bidirectional(lambda: LSTMCell(64), return_sequences=True, merge='concat'|'sum'|'avg'|'mul')."""

    def __init__(self, make_cell, return_sequences=False, merge="concat", name=None):
        """``make_cell``: a cell instance (cloned for the backward direction; serialisable)
        or a zero-arg factory (not serialisable)."""
        super().__init__(name)
        if isinstance(make_cell, Module):
            from .core import from_config
            cfg = make_cell.get_config()
            fwd_cell, bwd_cell = make_cell, from_config(cfg)
        else:
            fwd_cell, bwd_cell = make_cell(), make_cell()
        self.fwd = RNN(fwd_cell, return_sequences)
        self.bwd = RNN(bwd_cell, return_sequences, reverse=True)
        self.merge = merge

    def forward(self, x, mask=None, training=False):
        a, b = self.fwd(x, mask=mask, training=training), self.bwd(x, mask=mask, training=training)
        return {"concat": lambda: tf.concat([a, b], -1), "sum": lambda: a + b,
                "avg": lambda: (a + b) / 2, "mul": lambda: a * b}[self.merge]()
