import os, sys, tempfile, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import apexnet as an


def roundtrip(model, example, atol=1e-5, **kw):
    ref = tf.nest.map_structure(lambda t: t.numpy(), model(example, training=False, **kw))
    with tempfile.TemporaryDirectory() as d:
        an.save_model(model, d, example)
        json.load(open(d + "/config.json"))  # valid JSON
        m2 = an.load_model(d)
    out = tf.nest.map_structure(lambda t: t.numpy(), m2(example, training=False, **kw))
    tf.nest.map_structure(lambda a, b: np.testing.assert_allclose(a, b, atol=atol), ref, out)
    return m2


# ------------------------------------------------------------ save / load
def test_save_load_many_architectures():
    rs = np.random.RandomState(0)
    x = rs.randn(4, 10).astype("float32")
    tf.random.set_seed(0)
    m = an.Sequential(an.Dense(8, "gelu", regularizer=an.regularizers.l2(1e-3), init=an.core.normal(0.1)),
                      an.BatchNorm(), tf.nn.relu, an.Dropout(0.1), an.Dense(3))
    an.Trainer(m, an.cross_entropy(), an.Adam(1e-2)).fit(x, np.array([0, 1, 2, 0], "int32"), epochs=3, verbose=0)
    m2 = roundtrip(m, x)
    assert abs(float(m2.regularization_loss()) - float(m.regularization_loss())) < 1e-9  # regulariser survived
    ids = rs.randint(0, 20, (3, 9)).astype("int32")
    roundtrip(an.models.TransformerLM(20, 16, 2, 2, 12, rotary=True), ids)
    roundtrip(an.models.ResNet(4, (1, 1), (4, 8)), rs.randn(2, 8, 8, 3).astype("float32"))
    roundtrip(an.models.ViT(3, 8, 4, 8, 1, 2), rs.randn(2, 8, 8, 3).astype("float32"))
    roundtrip(an.models.UNet(2, 4, 2), rs.randn(2, 8, 8, 1).astype("float32"))
    roundtrip(an.models.BiLSTMClassifier(20, 2, 8, 8), ids)
    roundtrip(an.Sequential(an.Conv1D(4, 3, padding="causal"), an.LSTM(5), an.Dense(2)), rs.randn(2, 7, 3).astype("float32"))
    src = rs.randint(1, 9, (2, 5)).astype("int32"); tgt = rs.randint(1, 9, (2, 4)).astype("int32")
    roundtrip(an.models.Seq2SeqTransformer(9, 9, 16, 1, 2, 8), (src, tgt))  # tuple inputs


def test_save_load_custom_state_and_errors():
    texts = np.array(["red green", "blue red red"])
    tv = an.TextVectorizer(max_len=4).adapt(texts)
    m = an.Sequential(tv, an.Embedding(tv.vocab_size, 4), an.GlobalAvgPool(), an.Dense(2))
    m2 = roundtrip(m, tf.constant(texts))  # vocabulary travels in the config 'state'
    assert m2.layers[0].vocab == tv.vocab

    @an.register
    class Gated(an.Module):
        def __init__(self, units, act="sigmoid"):
            super().__init__(); self.units, self.d = units, an.Dense(units, act)
        def forward(self, x, training=False): return self.d(x) * 2.0
    g = roundtrip(Gated(3, act="tanh"), tf.ones([2, 5])); assert isinstance(g, Gated) and g.units == 3
    for bad in (an.Sequential(an.Lambda(lambda z: z * 2), an.Dense(1)), an.Wrap(tf.keras.layers.Dense(2))):
        try:
            bad.init(tf.ones([1, 3])); bad.get_config(); raise AssertionError("should not serialise")
        except ValueError as e:
            assert "serialise" in str(e) or "config" in str(e)
    an.register(lambda_fn := (lambda z: z + 1), name="plus_one")
    ok = roundtrip(an.Sequential(an.Lambda(lambda_fn), an.Dense(1)), tf.ones([2, 3])); assert ok is not None


def test_summary_shows_output_shapes():
    m = an.Sequential(an.Conv2D(4, 3), an.MaxPool2D(2), an.GlobalAvgPool(), an.Dense(5))
    txt = m.summary(np.zeros([2, 8, 8, 3], "float32"))
    assert "(2, 8, 8, 4)" in txt and "(2, 4, 4, 4)" in txt and "(2, 5)" in txt and "output shape" in txt
    assert "total params" in txt and "  conv2_d" in txt  # children are indented under the Sequential


# -------------------------------------------------------------------- SAM
def toy(n=128, seed=0):
    rs = np.random.RandomState(seed); x = rs.randn(n, 6).astype("float32")
    return x, (x[:, 0] * x[:, 1] > 0).astype("int32")


def test_sam_exact_restore_learns_and_nan_safe():
    x, y = toy()
    mk = lambda: an.Sequential(an.Dense(16, "tanh"), an.Dense(2))
    tf.random.set_seed(0); m = mk(); m.init(x[:2])
    tr = an.Trainer(m, an.cross_entropy(), an.SGD(0.0), sam=0.5)
    w0 = [v.numpy().copy() for v in m.variables]
    tr.fit(x, y, epochs=2, batch_size=32, verbose=0)
    for a, b in zip(w0, m.variables):
        np.testing.assert_array_equal(a, b.numpy())  # perturbation is undone bit-for-bit
    # learns, and takes a different path than plain SGD
    def run(sam):
        tf.random.set_seed(0); mm = mk(); mm.init(x[:2])
        t = an.Trainer(mm, an.cross_entropy(), an.SGD(0.2, 0.9), metrics=[an.Accuracy()], sam=sam)
        h = t.fit(x, y, epochs=40, batch_size=32, verbose=0); return h, [v.numpy() for v in mm.variables]
    hs, ws = run(0.05); hp, wp = run(None)
    assert hs["loss"][-1] < hs["loss"][0] * 0.7 and any(not np.allclose(a, b) for a, b in zip(ws, wp))
    # NaN batch: SAM must not poison weights
    bad = an.Trainer(an.Sequential(an.Dense(1)), lambda yy, p: tf.reduce_mean(p, -1) * tf.math.log(-tf.abs(tf.cast(yy, tf.float32)) - 1.0),
                     an.SGD(0.1), sam=0.1)
    xx = np.ones((8, 3), "float32"); yy = np.ones((8,), "int32"); bad.init(xx)
    before = bad.get_weights(); bad.fit(xx, yy, epochs=1, batch_size=4, verbose=0)
    for a, b in zip(before, bad.get_weights()):
        assert np.isfinite(b).all(); np.testing.assert_array_equal(a, b)


# --------------------------------------------- class_weight / validation_split
def test_class_weight_exact_and_validation_split():
    x, y = toy(96)
    tf.random.set_seed(0)
    m = an.Sequential(an.Dense(8, "tanh"), an.Dense(2)); m.init(x[:2])
    tr = an.Trainer(m, an.cross_entropy(), an.SGD(0.0), metrics=[an.Accuracy()])
    logits = m(x).numpy(); nll = -(logits - np.log(np.exp(logits).sum(1, keepdims=True)))[np.arange(len(y)), y]
    h = tr.fit(x, y, epochs=1, batch_size=96, class_weight={0: 1.0, 1: 0.0}, verbose=0)
    np.testing.assert_allclose(h["loss"][0], nll[y == 0].mean(), rtol=1e-5)            # arrays
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(32)
    tr2 = an.Trainer(m, an.cross_entropy(), an.SGD(0.0))
    h2 = tr2.fit(ds, epochs=1, class_weight={0: 1.0, 1: 3.0}, verbose=0)               # datasets
    w = np.where(y == 1, 3.0, 1.0); np.testing.assert_allclose(h2["loss"][0], (nll * w).sum() / w.sum(), rtol=1e-4)
    sw = np.random.RandomState(1).rand(96).astype("float32")
    h3 = an.Trainer(m, an.cross_entropy(), an.SGD(0.0)).fit(x, y, epochs=1, batch_size=96, sample_weight=sw, verbose=0)
    np.testing.assert_allclose(h3["loss"][0], (nll * sw).sum() / sw.sum(), rtol=1e-4)
    tr3 = an.Trainer(m, an.cross_entropy(), an.SGD(0.0))
    h4 = tr3.fit(x, y, epochs=1, batch_size=24, validation_split=0.25, verbose=0)
    k = 24
    ref = an.Trainer(m, an.cross_entropy(), an.SGD(0.0)).evaluate(x[-k:], y[-k:])
    np.testing.assert_allclose(h4["val_loss"][0], ref["loss"], rtol=1e-5)               # LAST 25% held out
    np.testing.assert_allclose(h4["loss"][0], nll[:-k].mean(), rtol=1e-5)               # and excluded from training
    try:
        tr3.fit(ds, epochs=1, validation_split=0.2, verbose=0); raise AssertionError
    except ValueError:
        pass


# ----------------------------------------------------------------- lr_find
def test_lr_find_restores_everything():
    x, y = toy(256)
    tf.random.set_seed(0)
    m = an.Sequential(an.Dense(16, "relu"), an.Dense(2))
    tr = an.Trainer(m, an.cross_entropy(), an.AdamW(an.warmup_cosine(1e-3, 5, 100)), metrics=[an.Accuracy()])
    tr.fit(x, y, epochs=1, batch_size=32, verbose=0)
    before = [v.numpy().copy() for v in m.variables]; it0 = int(tr.opt.iterations.numpy()); ep0 = tr.epoch
    slots0 = [s.numpy().copy() for s in tr.opt.m]
    r = tr.lr_find(x, y, batch_size=32, start=1e-6, end=50.0, steps=60)
    assert np.all(np.diff(r["lrs"]) > 0) and 1e-6 <= r["suggested"] <= 50.0
    assert len(r["lrs"]) < 60, "diverging LRs should stop the sweep early"
    for a, b in zip(before, m.variables): np.testing.assert_array_equal(a, b.numpy())
    for a, b in zip(slots0, tr.opt.m): np.testing.assert_array_equal(a, b.numpy())
    assert int(tr.opt.iterations.numpy()) == it0 and tr.epoch == ep0 and callable(tr.opt.lr)
    h = tr.fit(x, y, epochs=ep0 + 3, batch_size=32, verbose=0)  # trainer still fully functional afterwards
    assert h["loss"][-1] < h["loss"][0] + 1e-9 and np.isfinite(h["loss"]).all()


# --------------------------------------------------------------- Lookahead
def test_lookahead():
    x = np.random.RandomState(0).randn(128, 5).astype("float32"); y = x @ np.arange(1, 6, dtype="float32")[:, None]
    def run(opt):
        tf.random.set_seed(0); m = an.Sequential(an.Dense(8, "tanh"), an.Dense(1)); m.init(x[:2])
        t = an.Trainer(m, an.mse, opt); h = t.fit(x, y, epochs=10, batch_size=32, verbose=0)
        return h, [v.numpy() for v in m.variables], t
    _, w_plain, _ = run(an.SGD(0.02, 0.9))
    _, w_a1, _ = run(an.Lookahead(an.SGD(0.02, 0.9), k=3, alpha=1.0))   # alpha=1 -> identical to the inner optimizer
    for a, b in zip(w_plain, w_a1): np.testing.assert_allclose(a, b, atol=1e-6)
    h, _, t = run(an.Lookahead(an.Adam(0.03), k=5, alpha=0.5)); assert h["loss"][-1] < 0.5 * h["loss"][0]
    # the wrapped optimizer is checkpointed (slow weights included)
    with tempfile.TemporaryDirectory() as d:
        t.save(d + "/c"); assert len(t.opt.slow) == len(t.model.trainable_variables)


# ------------------------------------------------------------------- tuner
def test_tuner_random_and_halving():
    x, y = toy(256); xv, yv = toy(128, seed=5)
    space = {"lr": an.tune.LogUniform(1e-4, 1e2), "width": an.tune.Choice([8, 16]), "depth": an.tune.Int(1, 2)}
    def build(p):
        m = an.models.MLP([p["width"]] * p["depth"], 2, "tanh")
        return an.Trainer(m, an.cross_entropy(), an.Adam(p["lr"]), metrics=[an.Accuracy()])
    r = an.tune.search(build, space, x, y, val_data=(xv, yv), trials=6, epochs=8, method="random", verbose=0, fit_kwargs={"batch_size": 64})
    assert len(r["trials"]) == 6 and r["best_score"] == min(t["score"] for t in r["trials"])
    assert r["best_params"]["lr"] < 5.0                       # absurd learning rates lose
    r2 = an.tune.search(build, space, x, y, val_data=(xv, yv), trials=6, epochs=8, method="random", verbose=0, fit_kwargs={"batch_size": 64})
    assert [t["params"] for t in r["trials"]] == [t["params"] for t in r2["trials"]]  # seeded -> reproducible
    h = an.tune.search(build, space, x, y, val_data=(xv, yv), metric="val_acc", mode="max", trials=9, epochs=9,
                       method="halving", eta=3, verbose=0, fit_kwargs={"batch_size": 64})
    ep = sorted({t["epochs"] for t in h["trials"]}); assert len(ep) >= 2 and max(ep) == 9   # survivors got more budget
    assert sum(t["epochs"] == 9 for t in h["trials"]) < 9 and h["best_trainer"].epoch == 9


# --------------------------------------------------------------- callbacks
def test_backup_restore_and_lr_scheduler():
    x, y = toy(64)
    with tempfile.TemporaryDirectory() as d:
        mk = lambda: an.Trainer(an.models.MLP([8], 2), an.cross_entropy(), an.Adam(1e-2))
        t1 = mk(); t1.fit(x, y, epochs=2, batch_size=32, verbose=0, callbacks=[an.BackupAndRestore(d)])
        t2 = mk(); h = t2.fit(x, y, epochs=4, batch_size=32, verbose=0, callbacks=[an.BackupAndRestore(d)])
        assert t2.epoch == 4 and len(h["loss"]) == 2          # "crashed" after 2 epochs, resumed for the other 2
    scales = []
    t3 = mk(); t3.fit(x, y, epochs=3, batch_size=32, verbose=0, callbacks=[
        an.LearningRateScheduler(lambda e, s: 0.5 ** e), an.LambdaCallback(on_epoch_end=lambda t, e, l: scales.append(l["lr"]))])
    np.testing.assert_allclose(scales, [1e-2, 5e-3, 2.5e-3], rtol=1e-5)


# -------------------------------------------------- performance regression guard
def test_dense_is_not_pathologically_slow():
    """tf.einsum('...i,io->...o') once made Dense ~100x slower than matmul on CPU. Guard the ratio."""
    x = tf.random.normal([128, 256]); d = an.Dense(512); d.init(x); w = tf.Variable(tf.random.normal([256, 512]))
    def bench(fn, n=30):
        fn(); t = time.perf_counter()
        for _ in range(n): fn()
        return (time.perf_counter() - t) / n
    @tf.function
    def f_apexnet():
        with tf.GradientTape() as t: l = tf.reduce_sum(d(x) ** 2)
        return t.gradient(l, d.w)
    @tf.function
    def f_raw():
        with tf.GradientTape() as t: l = tf.reduce_sum(tf.matmul(x, w) ** 2)
        return t.gradient(l, w)
    assert bench(f_apexnet) < 12 * bench(f_raw) + 2e-3


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name, flush=True)
            except Exception:
                fails += 1; print("FAIL", name, flush=True); traceback.print_exc()
    sys.exit(fails)
