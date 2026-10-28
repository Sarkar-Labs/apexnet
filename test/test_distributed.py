"""Run in its own process: needs logical CPU devices configured before TF init."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf
cpus = tf.config.list_physical_devices("CPU")
tf.config.set_logical_device_configuration(cpus[0], [tf.config.LogicalDeviceConfiguration()] * 2)
import apexnet as an

rs = np.random.RandomState(0)
x = rs.randn(64, 6).astype("float32")
y = (rs.rand(64) > .5).astype("int32")
w = (rs.rand(64) > .3).astype("float32")  # uneven sample weights across replicas

def run(strategy):
    tf.random.set_seed(0)
    with (strategy.scope() if strategy else tf.distribute.get_strategy().scope()):
        m = an.models.MLP([8], 2)
    tr = an.Trainer(m, an.cross_entropy(), an.SGD(0.1), metrics=[an.Accuracy()], strategy=strategy)
    ds = tf.data.Dataset.from_tensor_slices((x, y, w)).batch(16)
    tr.fit(ds, epochs=1, verbose=0)  # init after seed
    return tr

# identical init: build, copy weights
base = an.Trainer(an.models.MLP([8], 2), an.cross_entropy(), an.SGD(0.1), metrics=[an.Accuracy()])
ds = tf.data.Dataset.from_tensor_slices((x, y, w)).batch(16)
base.init(x[:2]); w0 = base.get_weights()
st = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])
dist = an.Trainer(an.models.MLP([8], 2), an.cross_entropy(), an.SGD(0.1), metrics=[an.Accuracy()], strategy=st)
dist.init(x[:2]); dist.set_weights(w0)
rb = base.fit(ds, epochs=3, verbose=0); rd = dist.fit(ds, epochs=3, verbose=0)
for a, b in zip(base.get_weights(), dist.get_weights()):
    np.testing.assert_allclose(a, b, atol=1e-5)
np.testing.assert_allclose(rb["loss"], rd["loss"], atol=1e-5)
print("PASS distributed == single-device (weights and loss), replicas =", st.num_replicas_in_sync)


# ---- second scenario: regulariser, LSTM (tf.scan), vector metric, accumulation, clipping, Adam
xs = rs.randn(64, 7, 3).astype("float32"); ys = (xs.sum((1, 2)) > 0).astype("int32")
def mk(strategy=None):
    ctx = strategy.scope() if strategy else tf.distribute.get_strategy().scope()
    with ctx:
        return an.Trainer(an.Sequential(an.LSTM(6), an.Dense(2, regularizer=an.regularizers.l2(0.05))),
                          an.cross_entropy(), an.Adam(1e-2), metrics=[an.Accuracy(), an.MacroF1(2)],
                          grad_clip=1.0, accumulate=2, strategy=strategy)
b2, d2 = mk(), mk(st)
dsx = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(16)
b2.init(xs[:2]); d2.init(xs[:2]); d2.set_weights(b2.get_weights())
rb2, rd2 = b2.fit(dsx, epochs=3, verbose=0), d2.fit(dsx, epochs=3, verbose=0)
for a, b in zip(b2.get_weights(), d2.get_weights()):
    np.testing.assert_allclose(a, b, atol=2e-5)
for k in ("loss", "acc", "macro_f1"):
    np.testing.assert_allclose(rb2[k], rd2[k], atol=1e-5)
print("PASS distributed LSTM + L2 + MacroF1 + accumulation + clipping == single-device")


# ---- scenario 3: step counter + LR schedule + Adam bias correction + exact counts (no phantom batches)
xd = rs.randn(224, 16).astype("float32"); yd = (xd[:, :4].argmax(1)).astype("int32")
mk3 = lambda strat: an.Trainer(an.Sequential(an.Dense(16, "tanh"), an.Dense(4)), an.cross_entropy(),
                               an.Adam(an.warmup_cosine(3e-2, 3, 40)), metrics=[an.Accuracy()], strategy=strat, steps_per_call=4)
tf.random.set_seed(0)
for static in (False, True):
    ds3 = tf.data.Dataset.from_tensor_slices((xd, yd)).batch(32, drop_remainder=True)   # static shape, known cardinality
    b3, d3 = mk3(None), mk3(st)
    for t in (b3, d3): t.static_batches = static
    b3.init(xd[:32]); d3.init(xd[:32]); d3.set_weights(b3.get_weights())
    rb3, rd3 = b3.fit(ds3, epochs=3, verbose=0), d3.fit(ds3, epochs=3, verbose=0)
    assert int(b3.opt.iterations.numpy()) == int(d3.opt.iterations.numpy()) == 21, (b3.opt.iterations.numpy(), d3.opt.iterations.numpy())
    assert float(d3.train_meter.v["loss"][1].numpy()) == 224.0, "phantom batch leaked into the metrics"
    for a, b in zip(b3.get_weights(), d3.get_weights()):
        np.testing.assert_allclose(a, b, atol=2e-5)
    for k in ("loss", "acc", "lr"):
        np.testing.assert_allclose(rb3[k], rd3[k], rtol=1e-5, atol=1e-6)
print("PASS distributed Adam + warmup_cosine schedule: step counter, lr, metrics exact (static + dynamic)")

# ---- scenario 4: variables created in a Module's __init__ (outside the strategy scope) are adopted + distributed
def mk4(strat):
    return an.Trainer(an.models.TransformerLM(13, 16, 1, 2, 8), an.cross_entropy(), an.AdamW(1e-2, weight_decay=0.0),
                      strategy=strat, steps_per_call=2)
tokens = rs.randint(0, 13, (64, 8)).astype("int32"); tgt = np.roll(tokens, -1, 1)
ds4 = tf.data.Dataset.from_tensor_slices((tokens, tgt)).batch(16, drop_remainder=True)
b4, d4 = mk4(None), mk4(st)       # NB: both models were constructed OUTSIDE any scope (self.pos made in __init__)
b4.init(tokens[:16]); d4.init(tokens[:16])
assert all(st.extended.variable_created_in_scope(v) for v in d4.model.variables), "an unmirrored variable survived"
d4.set_weights(b4.get_weights())
b4.fit(ds4, epochs=2, verbose=0); d4.fit(ds4, epochs=2, verbose=0)
for a, b in zip(b4.get_weights(), d4.get_weights()):
    np.testing.assert_allclose(a, b, atol=3e-5)
print("PASS distributed: init-time variables (TransformerLM.pos) are adopted into the strategy and train identically")

# ---- scenario 5: predict / padded eval / divisibility errors on 2 replicas
tr5 = an.Trainer(an.Sequential(an.Dense(16, "relu"), an.Dense(4)), an.cross_entropy(), an.SGD(0.0),
                 metrics=[an.Accuracy(), an.MacroF1(4)], strategy=st, static_batches=True)
base5 = an.Trainer(an.Sequential(an.Dense(16, "relu"), an.Dense(4)), an.cross_entropy(), an.SGD(0.0), metrics=[an.Accuracy(), an.MacroF1(4)])
tr5.init(xd[:32]); base5.init(xd[:32]); tr5.set_weights(base5.get_weights())
xo, yo = xd[:83], yd[:83]                                      # 83 = not a multiple of the batch (or of the replicas)
pd, pb = tr5.predict(xo, batch_size=32), base5.predict(xo, batch_size=32)
assert pd.shape == (83, 4); np.testing.assert_allclose(pd, pb, atol=1e-5)
ed, eb = tr5.evaluate(xo, yo, batch_size=32), base5.evaluate(xo, yo, batch_size=32)
for k in ed: np.testing.assert_allclose(ed[k], eb[k], rtol=1e-5, atol=1e-6, err_msg=k)
try:
    tr5.evaluate(xo, yo, batch_size=33); raise AssertionError("indivisible batch accepted")
except ValueError as e:
    assert "divisible by the number of replicas" in str(e)
print("PASS distributed: predict + padded static eval equal single-device; indivisible batches rejected clearly")
