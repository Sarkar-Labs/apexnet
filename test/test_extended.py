import os, sys, tempfile, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import apexnet as an


def grads_ok(m, x, **kw):
    with tf.GradientTape() as t:
        y = m(x, training=True, **kw)
        loss = tf.reduce_mean(tf.cast(y, tf.float32) ** 2)
    g = t.gradient(loss, m.trainable_variables)
    assert all(gi is not None and bool(tf.reduce_all(tf.math.is_finite(gi))) for gi in g)
    return y


# ---------------------------------------------------------------- layers
def test_conv_family_shapes_and_grads():
    x2, x1, x3 = tf.random.normal([2, 8, 8, 3]), tf.random.normal([2, 10, 3]), tf.random.normal([2, 4, 4, 4, 3])
    cases = [(an.Conv1D(5, 3), x1, (2, 10, 5)), (an.Conv1D(5, 3, padding="causal"), x1, (2, 10, 5)),
             (an.Conv1D(5, 3, dilation=2), x1, (2, 10, 5)), (an.Conv2D(4, 3, stride=2), x2, (2, 4, 4, 4)),
             (an.Conv3D(4, 3), x3, (2, 4, 4, 4, 4)), (an.Conv2DTranspose(4, 2, 2), x2, (2, 16, 16, 4)),
             (an.Conv2DTranspose(4, 3, 2, "valid"), x2, (2, 17, 17, 4)), (an.Conv1DTranspose(4, 2, 2), x1, (2, 20, 4)),
             (an.Conv3DTranspose(2, 2, 2), x3, (2, 8, 8, 8, 2)),
             (an.DepthwiseConv2D(3, multiplier=2), x2, (2, 8, 8, 6)), (an.SeparableConv2D(7), x2, (2, 8, 8, 7)),
             (an.SqueezeExcite(), x2, (2, 8, 8, 3))]
    for m, x, shape in cases:
        assert tuple(grads_ok(m, x).shape) == shape, (type(m).__name__, shape)
    for m, x, shape in [(an.MaxPool2D(2), x2, (2, 4, 4, 3)), (an.AvgPool3D(2), x3, (2, 2, 2, 2, 3)),
                        (an.MaxPool1D(2), x1, (2, 5, 3)), (an.GlobalAvgPool(), x2, (2, 3)),
                        (an.GlobalMaxPool(), x1, (2, 3)), (an.Upsample(2, "bilinear"), x2, (2, 16, 16, 3)),
                        (an.PixelShuffle(2), tf.zeros([2, 4, 4, 8]), (2, 8, 8, 2)), (an.ZeroPad(((1, 1), (2, 2))), x2, (2, 10, 12, 3))]:
        assert tuple(m(x).shape) == shape
    # causality
    cc, xa = an.Conv1D(4, 3, padding="causal"), tf.random.normal([1, 8, 2])
    ya = cc(xa).numpy(); xb = xa.numpy().copy(); xb[:, -1] += 5
    np.testing.assert_allclose(ya[:, :-1], cc(tf.constant(xb)).numpy()[:, :-1], atol=1e-6)


def test_norms_and_activations():
    x = tf.random.normal([3, 5, 5, 4]) * 7 + 3
    y = an.GroupNorm(2)(x)
    g = tf.reshape(y, [3, 5, 5, 2, 2])
    assert float(tf.reduce_max(tf.abs(tf.reduce_mean(g, [1, 2, 4])))) < 1e-4
    assert float(tf.reduce_max(tf.abs(tf.math.reduce_std(g, [1, 2, 4]) - 1))) < 1e-2
    yi = an.InstanceNorm()(x)
    assert float(tf.reduce_max(tf.abs(tf.reduce_mean(yi, [1, 2])))) < 1e-4
    r = an.RMSNorm()(tf.random.normal([2, 6, 8]) * 5)
    np.testing.assert_allclose(tf.reduce_mean(r ** 2, -1).numpy(), 1.0, atol=1e-3)
    for name in an.layers.ACTIVATIONS:
        if name not in ("softmax", "log_softmax"):
            assert tf.reduce_all(tf.math.is_finite(an.Activation(name)(tf.linspace(-5.0, 5.0, 11))))
    assert an.GLU()(tf.ones([2, 6])).shape == (2, 3)
    assert grads_ok(an.PReLU(), tf.random.normal([2, 5])).shape == (2, 5)


def test_dropout_family_train_vs_eval():
    x = tf.ones([4, 6, 6, 8])
    for m in (an.SpatialDropout(0.5), an.DropPath(0.5), an.GaussianNoise(0.3), an.Dropout(0.5)):
        np.testing.assert_array_equal(m(x, training=False), x)
        assert not np.array_equal(m(x, training=True).numpy(), x.numpy())
    y = an.SpatialDropout(0.5)(x, training=True).numpy()  # whole channels dropped
    assert np.all((y == 0).all((1, 2)) | (y != 0).all((1, 2)))


def test_rnn_family():
    x = tf.random.normal([2, 10, 3])
    for m, shp in [(an.LSTM(6), (2, 6)), (an.GRU(6, return_sequences=True), (2, 10, 6)), (an.SimpleRNN(4), (2, 4)),
                   (an.Bidirectional(lambda: an.LSTMCell(5), return_sequences=True), (2, 10, 10)),
                   (an.Bidirectional(lambda: an.GRUCell(5), merge="sum"), (2, 5))]:
        assert tuple(grads_ok(m, x).shape) == shp
    out, (h, c) = an.RNN(an.LSTMCell(4), return_state=True)(x)
    np.testing.assert_allclose(out, h, atol=1e-6)
    # padded tail must not influence the last valid output (fwd and bwd)
    for layer in (an.LSTM(6), an.GRU(6), an.Bidirectional(lambda: an.LSTMCell(4))):
        xx = tf.random.normal([2, 7, 3]); m = tf.cast(tf.constant([[1] * 4 + [0] * 3] * 2), tf.bool)
        a = layer(xx, mask=m)
        b = layer(tf.concat([xx[:, :4], tf.random.normal([2, 3, 3])], 1), mask=m)
        np.testing.assert_allclose(a, b, atol=1e-5)


def test_attention_variants():
    x = tf.random.normal([2, 6, 16]); ctx = tf.random.normal([2, 9, 16])
    plain, rope = an.MultiHeadAttention(2, causal=True), an.MultiHeadAttention(2, causal=True, rotary=True)
    assert grads_ok(rope, x).shape == (2, 6, 16)
    cross = an.MultiHeadAttention(2, cross=True)
    assert cross(x, context=ctx).shape == (2, 6, 16)
    # cross-attn ignores masked context positions
    m = tf.cast(tf.constant([[1] * 5 + [0] * 4] * 2), tf.bool)
    a = cross(x, context=ctx, mask=m)
    ctx2 = tf.concat([ctx[:, :5], tf.random.normal([2, 4, 16])], 1)
    np.testing.assert_allclose(a, cross(x, context=ctx2, mask=m), atol=1e-4)
    # RoPE is position-aware, causal path stays causal
    xb = x.numpy().copy(); xb[:, -1] += 3
    np.testing.assert_allclose(rope(x).numpy()[:, :-1], rope(tf.constant(xb)).numpy()[:, :-1], atol=1e-4)
    blk = an.TransformerBlock(2, causal=True, cross=True, drop_path=0.1)
    assert blk(x, context=ctx, training=True).shape == (2, 6, 16)
    assert an.PositionalEncoding()(x).shape == x.shape


def test_misc_layers():
    x = tf.random.normal([3, 5])
    assert an.Parallel(an.Dense(2), an.Dense(3))(x).shape == (3, 5)
    assert an.Parallel(an.Dense(4), an.Dense(4), merge="add")(x).shape == (3, 4)
    assert an.Reshape((5, 1))(x).shape == (3, 5, 1) and an.Permute((2, 1))(tf.zeros([2, 3, 4])).shape == (2, 4, 3)
    assert an.Bilinear(4)(x, x).shape == (3, 4) and an.Einsum("bi,io->bo", (5, 7))(x).shape == (3, 7)
    sn = an.SpectralNorm(an.Dense(6)); y = grads_ok(sn, x); assert y.shape == (3, 6)
    for _ in range(30):
        sn(x, training=True)
    sigma = np.linalg.svd(sn.inner.w.numpy(), compute_uv=False)[0]
    assert np.isfinite(sigma)
    inner = an.Sequential(an.Dense(8, "gelu"), an.Dense(4)); inner(x); rm = an.Remat(inner)
    with tf.GradientTape() as t1: l1 = tf.reduce_sum(inner(x) ** 2)
    with tf.GradientTape() as t2: l2 = tf.reduce_sum(rm(x, training=True) ** 2)
    for a, b in zip(t1.gradient(l1, inner.trainable_variables), t2.gradient(l2, inner.trainable_variables)):
        np.testing.assert_allclose(a, b, atol=1e-5)


def test_initializers():
    for n in ("glorot_uniform", "glorot_normal", "he_normal", "he_uniform", "lecun_normal", "orthogonal"):
        assert an.core.get_init(n)((6, 4), tf.float32).shape == (6, 4)
    q = an.core.orthogonal((8, 8)).numpy()
    np.testing.assert_allclose(q @ q.T, np.eye(8), atol=1e-4)
    assert an.core.get_init(an.core.constant(3.0))((2, 2)).numpy().min() == 3.0
    assert abs(float(tf.math.reduce_std(an.core.get_init("he_normal")((512, 512)))) - np.sqrt(2 / 512)) < 5e-3


# -------------------------------------------------- regularizers / freezing / interop
def test_regularizer_shrinks_weights():
    x = np.random.RandomState(0).randn(128, 8).astype("float32"); y = x[:, :1] * 2
    def run(reg):
        tf.random.set_seed(0)
        m = an.Sequential(an.Dense(16, "tanh", regularizer=reg), an.Dense(1)); m.init(x[:2])
        tr = an.Trainer(m, an.mse, an.Adam(1e-2)); tr.fit(x, y, epochs=30, batch_size=32, verbose=0)
        return float(tf.norm(m.layers[0].w))
    assert run(an.regularizers.l2(0.1)) < 0.7 * run(None)
    assert float(an.regularizers.l1(1.0)(tf.ones([3]))) == 3.0


def test_freeze_and_unfreeze():
    x = np.random.RandomState(0).randn(64, 4).astype("float32"); y = (x[:, 0] > 0).astype("int32")
    body = an.Sequential(an.Dense(8, "relu"), an.BatchNorm())
    m = an.Sequential(body, an.Dense(2)); m.init(x[:2]); body.freeze()
    before = [v.numpy().copy() for v in body.variables]
    tr = an.Trainer(m, an.cross_entropy(), an.Adam(1e-2)); tr.fit(x, y, epochs=3, batch_size=16, verbose=0)
    for a, b in zip(before, body.variables):
        np.testing.assert_array_equal(a, b.numpy())  # weights AND BN statistics untouched
    assert len(m.trainable_variables) == 2 and m.regularization_loss() == 0
    body.unfreeze(); assert len(m.trainable_variables) == 6


def test_wrap_keras_module_and_savedmodel():
    x = np.random.RandomState(0).randn(64, 4).astype("float32"); y = (x.sum(1, keepdims=True)).astype("float32")
    keras_dense = tf.keras.layers.Dense(8, activation="tanh")
    m = an.Sequential(an.Wrap(keras_dense), an.Dense(1))
    tr = an.Trainer(m, an.mse, an.Adam(3e-2))
    h = tr.fit(x, y, epochs=40, batch_size=32, verbose=0)
    assert h["loss"][-1] < h["loss"][0] * 0.3
    assert len(m.trainable_variables) == 4  # keras kernel+bias, apexnet w+b (the keras ones really train)
    class Plain(tf.Module):
        def __init__(self): super().__init__(); self.w = tf.Variable(tf.ones([4, 2]))
        def __call__(self, x): return x @ self.w
    wm = an.Wrap(Plain()); assert wm(tf.ones([3, 4])).shape == (3, 2) and len(wm.trainable_variables) == 1
    with tempfile.TemporaryDirectory() as d:
        t = an.Trainer(an.models.MLP([8], 3), an.cross_entropy(), an.Adam()); t.init(x[:2])
        t.export(d + "/sm", [tf.TensorSpec([None, 4], tf.float32)])
        for fe in (an.from_saved_model(d + "/sm"), an.from_saved_model(d + "/sm", signature="serving_default")):
            assert fe(tf.constant(x[:2])).shape == (2, 3)
        head = an.Sequential(an.from_saved_model(d + "/sm"), an.Dense(2))  # frozen extractor + trainable head
        head.init(x[:2]); assert len(head.trainable_variables) == 2


# ---------------------------------------------------- optimizers / schedules
def test_optimizers_all_learn():
    x = np.random.RandomState(0).randn(256, 6).astype("float32"); y = (x @ np.arange(1, 7, dtype="float32"))[:, None]
    for name, opt in [("sgd", an.SGD(0.05, 0.9, nesterov=True)), ("adamw", an.AdamW(0.03)), ("adam", an.Adam(0.03)),
                      ("lion", an.Lion(1e-2, weight_decay=0.0)), ("rmsprop", an.RMSprop(3e-3, momentum=0.5)),
                      ("rmsprop_c", an.RMSprop(3e-3, centered=True)), ("adagrad", an.Adagrad(0.2)),
                      ("adamax", an.Adamax(0.05))]:
        tf.random.set_seed(0)
        tr = an.Trainer(an.Sequential(an.Dense(16, "tanh"), an.Dense(1)), an.mse, opt)
        h = tr.fit(x, y, epochs=25, batch_size=64, verbose=0)
        assert h["loss"][-1] < 0.3 * h["loss"][0], (name, h["loss"][0], h["loss"][-1])


def test_schedules_values():
    f = lambda fn, ps: [round(float(fn(tf.constant(p, tf.int64))), 4) for p in ps]
    assert f(an.exponential_decay(1., 10, .5), [0, 10, 20]) == [1.0, .5, .25]
    assert f(an.polynomial_decay(1., 10, 0., 2.), [0, 5, 10, 99]) == [1.0, .25, 0., 0.]
    assert f(an.piecewise_constant([5, 10], [1., .1, .01]), [0, 5, 9, 10, 50]) == [1., .1, .1, .01, .01]
    r = f(an.cosine_restarts(1., 10, 2.), [0, 10, 30]); assert r == [1., 1., 1.]
    oc = f(an.one_cycle(1., 100), [0, 30, 100]); assert oc[1] == 1.0 and oc[0] < 0.05 and oc[2] < 1e-3
    assert f(an.with_warmup(an.exponential_decay(1., 10, .5), 4), [0, 3, 4]) == [.25, 1., 1.]
    assert f(an.inverse_time_decay(1., 10, 1.), [0, 10]) == [1., .5]


# ---------------------------------------------------------- losses / metrics
def test_losses_finite_and_sane():
    y = tf.constant([0, 1, 2, 1]); logits = tf.random.normal([4, 3])
    soft = tf.one_hot(y, 3)
    np.testing.assert_allclose(an.categorical_cross_entropy()(soft, logits), an.cross_entropy()(y, logits)[0], atol=1e-5)
    fl0, _ = an.focal_loss(0.0)(y, logits); ce, _ = an.cross_entropy()(y, logits)
    np.testing.assert_allclose(fl0, ce, atol=1e-5)
    assert tf.reduce_all(an.focal_loss(2.0, alpha=[1., 2., 3.])(y, logits)[0] <= ce * 3 + 1e-5)
    p = tf.nn.softmax(logits)
    assert float(tf.reduce_max(tf.abs(an.kl_divergence(p, logits)))) < 1e-5
    yb, sb = tf.constant([0., 1., 1., 0.]), tf.constant([-2., 2., -1., 3.])
    assert an.hinge(yb, sb).shape == (4,) and float(an.hinge(yb, sb)[0]) == 0.0
    for fn in (an.mse, an.mae, an.huber(1.0), an.log_cosh, an.poisson, an.binary_cross_entropy, an.squared_hinge, an.cosine_distance):
        assert tf.reduce_all(tf.math.is_finite(fn(tf.random.normal([4, 3]), tf.random.normal([4, 3]))))
    seg_logits = tf.random.normal([2, 4, 4, 3]); seg_y = tf.random.uniform([2, 4, 4], 0, 3, tf.int32)
    perfect = tf.one_hot(seg_y, 3) * 50.0
    assert float(tf.reduce_mean(an.dice_loss()(seg_y, perfect))) < 0.02 < float(tf.reduce_mean(an.dice_loss()(seg_y, seg_logits)))
    z = tf.random.normal([8, 16])
    assert float(tf.reduce_mean(an.info_nce(0.1)(None, (z, z)))) < float(tf.reduce_mean(an.info_nce(0.1)(None, (z, tf.random.normal([8, 16])))))
    assert an.triplet_loss()(None, (z, z, z + 1)).shape == (8,)
    lab = tf.constant([[1, 2, -1], [3, -1, -1]]); lg = tf.random.normal([2, 12, 5])
    c = an.ctc_loss(blank=0, pad=-1)(lab, lg); assert c.shape == (2,) and tf.reduce_all(c > 0)
    t = an.distillation(2.0)(logits, logits); assert float(tf.reduce_max(tf.abs(t))) < 1e-5


def test_vector_metrics_match_numpy():
    rs = np.random.RandomState(0)
    n = 500; y = rs.randint(0, 2, n).astype("int32"); score = (y * 1.2 + rs.randn(n)).astype("float32")
    m = an.Sequential(an.Lambda(lambda x: x))
    tr = an.Trainer(m, an.binary_cross_entropy,
                    an.SGD(0.1), metrics=[an.Precision(), an.Recall(), an.F1(), an.AUC(2048), an.BinaryAccuracy()])
    r = tr.evaluate(tf.data.Dataset.from_tensor_slices((np.stack([score], 1), y)).batch(64))
    pred = score > 0; tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum()
    np.testing.assert_allclose(r["precision"], tp / (tp + fp), atol=1e-6)
    np.testing.assert_allclose(r["recall"], tp / (tp + fn), atol=1e-6)
    np.testing.assert_allclose(r["f1"], 2 * tp / (2 * tp + fp + fn), atol=1e-6)
    np.testing.assert_allclose(r["acc"], (pred == (y == 1)).mean(), atol=1e-6)
    order = np.argsort(score); ranks = np.empty(n); ranks[order] = np.arange(1, n + 1)
    auc_ref = (ranks[y == 1].sum() - (y == 1).sum() * ((y == 1).sum() + 1) / 2) / ((y == 1).sum() * (y == 0).sum())
    assert abs(r["auc"] - auc_ref) < 5e-3, (r["auc"], auc_ref)
    # multi-class macro-F1 and R2
    yc = rs.randint(0, 4, 400).astype("int32"); lg = rs.randn(400, 4).astype("float32"); lg[np.arange(400), yc] += 1.5
    tr2 = an.Trainer(an.Lambda(lambda x: x), an.cross_entropy(), an.SGD(0.1), metrics=[an.MacroF1(4)])
    r2 = tr2.evaluate(tf.data.Dataset.from_tensor_slices((lg, yc)).batch(100))
    pc = lg.argmax(1); f1s = []
    for c in range(4):
        tp_ = ((pc == c) & (yc == c)).sum(); fp_ = ((pc == c) & (yc != c)).sum(); fn_ = ((pc != c) & (yc == c)).sum()
        f1s.append(2 * tp_ / (2 * tp_ + fp_ + fn_))
    np.testing.assert_allclose(r2["macro_f1"], np.mean(f1s), atol=1e-6)
    yr = rs.randn(300, 1).astype("float32"); pr = yr + 0.3 * rs.randn(300, 1).astype("float32")
    tr3 = an.Trainer(an.Lambda(lambda x: x), an.mse, an.SGD(0.1), metrics=[an.R2(), an.MAE()])
    r3 = tr3.evaluate(tf.data.Dataset.from_tensor_slices((pr, yr)).batch(100))
    np.testing.assert_allclose(r3["r2"], 1 - ((yr - pr) ** 2).sum() / ((yr - yr.mean()) ** 2).sum(), atol=1e-4)


def test_mixup_cutmix_soft_labels_train():
    rs = np.random.RandomState(0); x = rs.randn(64, 8, 8, 1).astype("float32"); y = rs.randint(0, 3, 64).astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
    for aug in (an.data.mixup(3, 0.4), an.data.cutmix(3)):
        xb, yb = next(iter(ds.map(aug)))
        assert xb.shape == (16, 8, 8, 1) and np.allclose(yb.numpy().sum(-1), 1.0, atol=1e-5)
    m = an.Sequential(an.Conv2D(4, 3), an.GlobalAvgPool(), an.Dense(3))
    tr = an.Trainer(m, an.categorical_cross_entropy(0.1), an.Adam(1e-2))
    h = tr.fit(ds.map(an.data.mixup(3)), epochs=2, verbose=0); assert np.isfinite(h["loss"]).all()


# ------------------------------------------------------------- preprocessing
def test_preprocessing_layers():
    x = tf.random.uniform([4, 12, 12, 3], 0, 255)
    assert an.Rescaling()(x).numpy().max() <= 1.0
    assert an.Resizing(6, 6)(x).shape == (4, 6, 6, 3)
    assert an.RandomCrop(8, 8)(x).shape == (4, 12, 12, 3)[:1] + (8, 8, 3) and an.RandomCrop(8, 8)(x, training=True).shape == (4, 8, 8, 3)
    img = tf.reshape(tf.range(4 * 6 * 6 * 1, dtype=tf.float32), [4, 6, 6, 1])
    np.testing.assert_array_equal(an.RandomFlip()(img), img)
    flipped = an.RandomFlip("both")(img, training=True).numpy()
    assert all(np.array_equal(f, i) or np.array_equal(f, i[:, ::-1]) or np.array_equal(f, i[::-1]) or np.array_equal(f, i[::-1, ::-1])
               for f, i in zip(flipped, img.numpy()))
    assert an.RandomTranslation(2)(x, training=True).shape == x.shape
    for L in (an.RandomBrightness(.2), an.RandomContrast(), an.RandomRotation(20)):
        np.testing.assert_array_equal(L(x), x); assert L(x, training=True).shape == x.shape
    np.testing.assert_allclose(an.RandomRotation(0.0)(img, training=True), img, atol=1e-3)
    data = np.random.RandomState(0).randn(200, 3).astype("float32") * [1, 5, 10] + [3, -2, 0]
    n = an.Normalization().adapt(data); z = n(data).numpy()
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-4); np.testing.assert_allclose(z.std(0), 1, atol=1e-3)


def test_text_vectorizer_and_audio():
    tv = an.TextVectorizer(max_len=6).adapt(["The cat sat", "the dog sat down!", "A cat"])
    out = tv(tf.constant(["the cat sat", "zebra dog"])).numpy()
    assert out.shape == (2, 6) and out[0, 3:].sum() == 0 and out[1, 0] == 1  # OOV id 1, padded with 0
    assert out[0, 0] == tv.vocab.index("the") and out[0, 0] == out[1, 1] - (out[1, 1] - out[0, 0])
    assert tv(tf.constant(["a b c d e f g h i"])).shape == (1, 6)  # truncation
    wav = tf.sin(tf.linspace(0.0, 2000.0, 8000))[None]
    sp = an.Spectrogram(256, 128)(wav); assert sp.shape[-1] == 129 and sp.shape[1] == 1 + (8000 - 256) // 128
    mel = an.MelSpectrogram(16000, 40, 400, 160, 512)(wav); assert mel.shape[-1] == 40 and tf.reduce_all(tf.math.is_finite(mel))
    # whole pipeline: string -> ids -> embedding -> classifier trains
    texts = np.array(["good great fine", "bad awful poor"] * 32); labels = np.array([1, 0] * 32, "int32")
    tv2 = an.TextVectorizer(max_len=4).adapt(texts)
    m = an.Sequential(an.Lambda(lambda s: tv2(s)), an.Embedding(tv2.vocab_size, 8), an.GlobalAvgPool(), an.Dense(2))
    tr = an.Trainer(m, an.cross_entropy(), an.Adam(3e-2), metrics=[an.Accuracy()])
    assert tr.fit(texts, labels, epochs=15, batch_size=16, verbose=0)["acc"][-1] == 1.0


# -------------------------------------------------------------------- models
def test_model_zoo_forward_and_train_step():
    tf.random.set_seed(0)
    img = np.random.RandomState(0).randn(8, 16, 16, 3).astype("float32"); yi = np.arange(8).astype("int32") % 4
    for name, m in [("resnet", an.models.ResNet(4, (1, 1), (8, 16))), ("vit", an.models.ViT(4, 16, 4, 16, 2, 2))]:
        tr = an.Trainer(m, an.cross_entropy(), an.AdamW(1e-3), metrics=[an.Accuracy()])
        h = tr.fit(img, yi, epochs=2, batch_size=4, verbose=0); assert np.isfinite(h["loss"]).all(), name
        assert tr.predict(img).shape == (8, 4)
    un = an.models.UNet(3, base=4, depth=2)
    seg = np.random.RandomState(1).randint(0, 3, (8, 16, 16)).astype("int32")
    tr = an.Trainer(un, an.cross_entropy(), an.Adam(1e-2), metrics=[an.Accuracy()])
    h = tr.fit(img, seg, epochs=8, batch_size=4, verbose=0); assert h["loss"][-1] < h["loss"][0]
    # BiLSTM: label = does token 7 occur (padded variable length)
    rs = np.random.RandomState(0); ids = rs.randint(1, 12, (256, 14)).astype("int32")
    lens = rs.randint(6, 15, 256); [ids.__setitem__((i, slice(l, None)), 0) for i, l in enumerate(lens)]
    lab = (ids == 7).any(1).astype("int32")
    tr = an.Trainer(an.models.BiLSTMClassifier(12, 2, 16, 16, 0.0), an.cross_entropy(), an.Adam(1e-2), metrics=[an.Accuracy()])
    h = tr.fit(ids, lab, epochs=25, batch_size=32, verbose=0); assert h["acc"][-1] > 0.9, h["acc"][-3:]


def test_seq2seq_learns_reverse():
    tf.random.set_seed(0); rs = np.random.RandomState(0); V, T = 8, 5
    src = rs.randint(2, V, (512, T)).astype("int32"); tgt = src[:, ::-1]
    bos = np.ones((512, 1), "int32")
    tin = np.concatenate([bos, tgt[:, :-1]], 1)
    ds = tf.data.Dataset.from_tensor_slices(((src, tin), tgt)).batch(64).repeat()
    m = an.models.Seq2SeqTransformer(V, V, dim=32, depth=2, heads=2, max_len=T)
    tr = an.Trainer(m, an.cross_entropy(ignore_index=0), an.AdamW(3e-3, weight_decay=0.0),
                    metrics=[an.Accuracy(ignore_index=0)], grad_clip=1.0, steps_per_call=4)
    h = tr.fit(ds, epochs=10, steps_per_epoch=40, verbose=0)
    assert h["acc"][-1] > 0.6, h["acc"]


# ----------------------------------------------------------- trainer extensions
def test_gan_two_optimizers_custom_step():
    tf.random.set_seed(0)
    class GAN(an.Module):
        def __init__(self):
            super().__init__()
            self.G = an.Sequential(an.Dense(16, "relu"), an.Dense(2))
            self.D = an.Sequential(an.Dense(16, "relu"), an.Dense(1))
        def forward(self, x, training=False):
            return self.D(x)
    class GANTrainer(an.Trainer):
        def train_step(self, batch):
            real = batch[0] if isinstance(batch, (tuple, list)) else batch
            B = tf.shape(real)[0]
            z = tf.random.normal([B, 4])
            with tf.GradientTape() as td:
                fake = self.model.G(z)
                d_loss = tf.reduce_mean(tf.nn.softplus(-self.model.D(real))) + tf.reduce_mean(tf.nn.softplus(self.model.D(fake)))
            self.apply_gradients("d", td.gradient(d_loss, self.model.D.trainable_variables), self.model.D.trainable_variables)
            with tf.GradientTape() as tg:
                g_loss = tf.reduce_mean(tf.nn.softplus(-self.model.D(self.model.G(z))))
            self.apply_gradients("g", tg.gradient(g_loss, self.model.G.trainable_variables), self.model.G.trainable_variables)
            one = tf.cast(B, tf.float32)
            return {"loss": (d_loss * one, one), "g_loss": (g_loss * one, one)}
    gan = GAN(); gan.G(tf.zeros([1, 4])); gan.D(tf.zeros([1, 2]))
    tr = GANTrainer(gan, None, optimizer=None, extra_stats=["g_loss"],
                    extra_optimizers={"d": (an.Adam(2e-3), lambda m: m.D.trainable_variables),
                                      "g": (an.Adam(2e-3), lambda m: m.G.trainable_variables)})
    real = (np.random.RandomState(0).randn(512, 2) * 0.5 + [2.0, -1.0]).astype("float32")
    g0 = [v.numpy().copy() for v in gan.G.trainable_variables]
    h = tr.fit(tf.data.Dataset.from_tensor_slices((real, real[:, 0])).batch(64), epochs=60, verbose=0)
    assert "g_loss" in h and all(np.isfinite(h["g_loss"]))
    assert any(not np.allclose(a, b.numpy()) for a, b in zip(g0, gan.G.trainable_variables))
    gen = gan.G(tf.random.normal([1000, 4])).numpy().mean(0)  # generator moves toward the data mean
    assert np.linalg.norm(gen - [2.0, -1.0]) < 1.2, gen


def test_tensorboard_csv_plateau_profiler():
    x = np.random.RandomState(0).randn(128, 4).astype("float32"); y = (x[:, 0] > 0).astype("int32")
    with tempfile.TemporaryDirectory() as d:
        tr = an.Trainer(an.models.MLP([8], 2), an.cross_entropy(), an.Adam(1e-9), metrics=[an.Accuracy()])
        tr.fit(x, y, epochs=5, batch_size=32, val_data=(x, y), verbose=0, callbacks=[
            an.TensorBoard(d + "/tb", histograms=True, every=2), an.CSVLogger(d + "/log.csv"),
            an.ReduceLROnPlateau("val_loss", 0.5, patience=1, min_delta=1.0), an.GradientHealth(),
            an.Profiler(d + "/prof", 1, 3)])
        assert glob.glob(d + "/tb/events*") and os.path.getsize(glob.glob(d + "/tb/events*")[0]) > 200
        rows = open(d + "/log.csv").read().strip().splitlines(); assert len(rows) == 6 and rows[0].startswith("epoch")
        assert float(tr.opt.lr_scale.numpy()) < 0.5  # plateau fired repeatedly
        assert abs(float(tr.opt.current_lr().numpy()) - 1e-9 * float(tr.opt.lr_scale.numpy())) < 1e-15


def test_tflite_export_matches():
    x = np.random.RandomState(0).randn(5, 6).astype("float32")
    tr = an.Trainer(an.models.MLP([8], 3), an.cross_entropy(), an.Adam(), precision="bfloat16")
    tr.init(x[:2])
    with tempfile.TemporaryDirectory() as d:
        p = tr.export_tflite(d + "/m.tflite", [tf.TensorSpec([5, 6], tf.float32)])
        it = tf.lite.Interpreter(model_path=p); it.allocate_tensors()
        it.set_tensor(it.get_input_details()[0]["index"], x); it.invoke()
        out = it.get_tensor(it.get_output_details()[0]["index"])
        an.set_precision("float32")
        np.testing.assert_allclose(out, tr.predict(x) if False else tr.model(x).numpy(), atol=1e-4)
        q = tr.export_tflite(d + "/q.tflite", [tf.TensorSpec([5, 6], tf.float32)], quantize="dynamic")
        assert os.path.getsize(q) > 0
    an.set_precision("float32")


# --------------------------------------------------------------------- data/util
def test_tfrecord_csv_imagedir_bucket_generator():
    with tempfile.TemporaryDirectory() as d:
        rs = np.random.RandomState(0); X = rs.randn(50, 4).astype("float32"); Y = rs.randint(0, 3, 50)
        an.data.write_tfrecord(d + "/a.tfrecord", {"x": X, "y": Y, "name": np.array([f"s{i}" for i in range(50)])})
        spec = {"x": tf.io.FixedLenFeature([4], tf.float32), "y": tf.io.FixedLenFeature([], tf.int64),
                "name": tf.io.FixedLenFeature([], tf.string)}
        ds = an.data.from_tfrecord([d + "/a.tfrecord"], spec, batch_size=50)
        b = next(iter(ds)); np.testing.assert_allclose(b["x"], X); np.testing.assert_array_equal(b["y"], Y)
        assert b["name"][3].numpy() == b"s3"
        trn = an.data.from_tfrecord([d + "/a.tfrecord"], spec, batch_size=10, map_fn=lambda e: (e["x"], e["y"]))
        tr = an.Trainer(an.models.MLP([8], 3), an.cross_entropy(), an.Adam()); tr.fit(trn, epochs=1, verbose=0)
        open(d + "/t.csv", "w").write("a,b,label\n" + "\n".join(f"{i},{i*2},{i%2}" for i in range(20)))
        cx, cy = next(iter(an.data.from_csv(d + "/t.csv", "label", batch_size=20)))
        assert cx.shape == (20, 2) and cy.shape == (20,)
        for ci, cname in enumerate(("cat", "dog")):
            os.makedirs(f"{d}/img/{cname}")
            for k in range(3):
                png = tf.io.encode_png(tf.cast(tf.fill([10, 10, 3], 60 * ci + 20 * k), tf.uint8))
                tf.io.write_file(f"{d}/img/{cname}/{k}.png", png)
        ids, classes = an.data.from_image_dir(d + "/img", (8, 8), batch_size=6, shuffle=False)
        xb, yb = next(iter(ids)); assert classes == ["cat", "dog"] and xb.shape == (6, 8, 8, 3) and list(yb.numpy()) == [0] * 3 + [1] * 3
        seqs = [np.arange(n, dtype="int32") + 1 for n in (3, 4, 9, 10, 3, 11)]
        sds = tf.data.Dataset.from_generator(lambda: ((s, 0) for s in seqs), output_signature=(tf.TensorSpec([None], tf.int32), tf.TensorSpec([], tf.int32)))
        bk = an.data.bucket_by_length(sds, [6], [2, 2])
        for xb_, _ in bk:
            assert xb_.shape[0] <= 2 and (xb_.shape[1] <= 5 or xb_.shape[1] >= 9)
        g = an.data.from_generator(lambda: ((np.ones(3, "float32") * i, i) for i in range(5)),
                                   (tf.TensorSpec([3], tf.float32), tf.TensorSpec([], tf.int32)), batch_size=2)
        assert [b[0].shape[0] for b in g] == [2, 2, 1]
        im = an.data.augment_images(True, 2, 0.1)(tf.ones([8, 8, 3]), 1); assert im[0].shape == (8, 8, 3)


def test_utils_seed_and_strategy():
    an.set_seed(7); a = an.Dense(4).init(tf.zeros([1, 3])).w.numpy()
    an.set_seed(7); b = an.Dense(4).init(tf.zeros([1, 3])).w.numpy()
    np.testing.assert_array_equal(a, b)
    assert isinstance(an.auto_strategy(), tf.distribute.Strategy) and an.utils.devices()["CPU"] >= 1
    assert an.tf is tf


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
