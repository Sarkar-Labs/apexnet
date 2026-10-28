hereimport os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import apexnet as an


def spirals(n=600, seed=0):
    rs = np.random.RandomState(seed)
    t = np.sqrt(rs.rand(n)) * 3 * np.pi
    c = rs.randint(0, 2, n)
    r = t + c * np.pi
    x = np.stack([r * np.cos(t), r * np.sin(t)], 1) / 10 + rs.randn(n, 2) * 0.01
    return x.astype("float32"), c.astype("int32")


def test_mlp_learns_and_summary():
    tf.random.set_seed(0)
    x, y = spirals()
    m = an.models.MLP([64, 64], 2, "gelu")
    tr = an.Trainer(m, an.cross_entropy(), an.AdamW(an.warmup_cosine(1e-2, 20, 1800)),
                    metrics=[an.Accuracy()], grad_clip=1.0)
    h = tr.fit(x, y, epochs=200, batch_size=64, verbose=0)
    assert h["loss"][-1] < h["loss"][0] * 0.5 and h["acc"][-1] > 0.9, (h["loss"][-1], h["acc"][-1])
    assert tr.evaluate(x, y)["acc"] > 0.9
    assert tr.predict(x[:7]).shape == (7, 2)
    assert "total params" in m.summary()


def test_cnn_batchnorm():
    tf.random.set_seed(0)
    rs = np.random.RandomState(0)
    x = rs.randn(256, 8, 8, 1).astype("float32")
    y = (x.mean((1, 2, 3)) > 0).astype("int32")
    m = an.Sequential(an.Conv2D(8, 3), an.BatchNorm(), an.Activation("relu"),
                      an.MaxPool2D(2), an.GlobalAvgPool2D(), an.Dense(2))
    tr = an.Trainer(m, an.cross_entropy(), an.AdamW(3e-3), metrics=[an.Accuracy()])
    h = tr.fit(x, y, epochs=30, batch_size=32, verbose=0)
    assert h["acc"][-1] > 0.9, h["acc"][-5:]
    assert float(tf.reduce_max(tf.abs(m.layers[1].mean))) > 0  # BN stats updated


def test_gradient_accumulation_equivalence():
    rs = np.random.RandomState(1)
    x, y = rs.randn(64, 5).astype("float32"), rs.randn(64, 1).astype("float32")

    def run(batch, acc):
        tf.random.set_seed(0)
        m = an.Sequential(an.Dense(8, "tanh"), an.Dense(1))
        tr = an.Trainer(m, an.mse, an.SGD(0.1, momentum=0.9), accumulate=acc)
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch)
        tr.init(x[:2])
        tr.fit(ds, epochs=3, verbose=0)
        assert int(tr.micro.numpy()) == 0 and int(tr.opt.iterations.numpy()) == 6
        return tr.get_weights()
    a, b = run(32, 1), run(16, 2)
    for p, q in zip(a, b):
        np.testing.assert_allclose(p, q, atol=1e-5)


def test_masked_loss_normalisation():
    # padding tokens must not contribute to loss
    y = tf.constant([[1, 2, -100], [3, -100, -100]])
    logits = tf.random.normal([2, 3, 5])
    lv, lw = an.cross_entropy(ignore_index=-100)(y, logits)
    assert float(tf.reduce_sum(lw)) == 3.0 and float(tf.reduce_sum(lv * lw)) > 0


def test_transformer_causal_and_learns():
    tf.random.set_seed(0)
    V, T = 16, 12
    lm = an.models.TransformerLM(V, dim=32, depth=2, heads=2, max_len=T)
    ids = tf.random.uniform([2, T], 0, V, tf.int32)
    lm.init(ids)
    base = lm(ids).numpy()
    ids2 = ids.numpy().copy(); ids2[:, -1] = (ids2[:, -1] + 1) % V
    np.testing.assert_allclose(base[:, :-1], lm(tf.constant(ids2)).numpy()[:, :-1], atol=1e-4)  # causal
    tokens = np.tile(np.arange(V), 400).astype("int32")  # deterministic cycle
    ds = an.data.lm_windows(tokens, T, 16, seed=0)
    tr = an.Trainer(lm, an.cross_entropy(), an.AdamW(3e-3, weight_decay=0.0),
                    metrics=[an.Perplexity(), an.Accuracy()], grad_clip=1.0, steps_per_call=4)
    h = tr.fit(ds, epochs=6, verbose=0)
    assert h["acc"][-1] > 0.9 and h["ppl"][-1] < 1.5, (h["acc"][-1], h["ppl"][-1])


def test_nan_step_is_skipped():
    m = an.Sequential(an.Dense(1))
    tr = an.Trainer(m, lambda y, p: tf.reduce_mean(p, -1) * tf.math.log(y[:, 0]), an.SGD(0.1))
    x = np.ones((8, 3), "float32")
    y = -np.ones((8, 1), "float32")  # log(-1) = nan
    tr.init(x)
    before = tr.get_weights()
    tr.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(4), epochs=1, verbose=0)
    for a, b in zip(before, tr.get_weights()):
        np.testing.assert_array_equal(a, b)
    assert int(tr.skipped.numpy()) == 2


def test_checkpoint_resume_and_export():
    x, y = spirals(256)
    mk = lambda: an.Trainer(an.models.MLP([32], 2), an.cross_entropy(), an.AdamW(1e-2),
                            metrics=[an.Accuracy()], ema=0.99)
    with tempfile.TemporaryDirectory() as d:
        t1 = mk()
        t1.fit(x, y, epochs=3, batch_size=64, verbose=0, callbacks=[an.Checkpoint(d, keep=2)])
        assert sorted(f for f in os.listdir(d) if f.endswith(".index")) == ["epoch_2.index", "epoch_3.index"]
        t2 = mk()
        t2.load(os.path.join(d, "epoch_3"))
        t2.fit(x, y, epochs=4, batch_size=64, verbose=0)  # resumes: only epoch 4 runs
        assert len(t2.history["loss"]) == 1 and t2.epoch == 4
        t3 = mk(); t3.load(os.path.join(d, "epoch_3")); t3.init(x[:2])
        np.testing.assert_allclose(t1.predict(x[:5]), t3.predict(x[:5]), atol=1e-5)
        path = t1.export(os.path.join(d, "sm"), [tf.TensorSpec([None, 2], tf.float32)])
        loaded = tf.saved_model.load(path)
        out = loaded.serve(tf.constant(x[:5]))
        np.testing.assert_allclose(out.numpy(), t1.predict(x[:5]), atol=1e-4)


def test_early_stopping_and_val():
    x, y = spirals(300)
    tr = an.Trainer(an.models.MLP([16], 2), an.cross_entropy(), an.AdamW(1e-9), metrics=[an.Accuracy()])
    h = tr.fit(x, y, epochs=50, batch_size=64, val_data=(x[:50], y[:50]), verbose=0,
               callbacks=[an.EarlyStopping("val_loss", patience=2, min_delta=1e-2)])
    assert len(h["loss"]) < 10 and "val_acc" in h


def test_ema_changes_eval_weights():
    x, y = spirals(256)
    tr = an.Trainer(an.models.MLP([16], 2), an.cross_entropy(), an.AdamW(1e-2), ema=0.9)
    tr.fit(x, y, epochs=2, batch_size=64, verbose=0)
    assert tr.evaluate(x, y, use_ema=True)["loss"] != tr.evaluate(x, y, use_ema=False)["loss"]


def test_bfloat16_and_float16():
    x, y = spirals(256)
    for prec in ("bfloat16", "float16"):
        tr = an.Trainer(an.models.MLP([32], 2), an.cross_entropy(), an.AdamW(1e-2),
                        metrics=[an.Accuracy()], precision=prec)
        h = tr.fit(x, y, epochs=8, batch_size=64, verbose=0)
        assert h["loss"][-1] < h["loss"][0], prec
        assert tr.predict(x[:3]).dtype == np.float32
    an.set_precision("float32")


def test_jit_and_custom_train_step():
    x, y = spirals(128)

    class MyTrainer(an.Trainer):
        calls = 0
        def train_step(self, batch):
            MyTrainer.calls += 1
            return super().train_step(batch)
    tr = MyTrainer(an.models.MLP([16], 2), an.cross_entropy(), an.Lion(1e-3), jit=True)
    tr.fit(x, y, epochs=1, batch_size=32, verbose=0)
    assert MyTrainer.calls >= 1


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception:
                fails += 1; print("FAIL", name); traceback.print_exc()
    sys.exit(fails)
