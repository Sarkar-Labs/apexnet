import os, sys, tempfile, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import torch
import jax, jax.numpy as jnp
import apexnet as an


def data(n=64, d=5, k=3, seed=0):
    rs = np.random.RandomState(seed); x = rs.randn(n, d).astype("float32")
    return x, (x[:, :k].argmax(1)).astype("int32")


def batches(x, y, bs):
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(bs, drop_remainder=True)


# ================================================================= PyTorch
def test_torch_training_matches_native_torch():
    x, y = data(64)
    torch.manual_seed(0)
    mk = lambda: torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
    fm = mk(); ref = mk(); ref.load_state_dict(fm.state_dict())
    tr = an.Trainer(an.Sequential(an.Torch(fm)), an.cross_entropy(), an.SGD(0.1, momentum=0.9), metrics=[an.Accuracy()])
    h = tr.fit(batches(x, y, 16), epochs=3, verbose=0)
    # the very same training, natively in PyTorch
    opt = torch.optim.SGD(ref.parameters(), lr=0.1, momentum=0.9); losses = []
    for _ in range(3):
        tot = 0.0
        for i in range(0, 64, 16):
            xb, yb = torch.from_numpy(x[i:i + 16]), torch.from_numpy(y[i:i + 16]).long()
            loss = torch.nn.functional.cross_entropy(ref(xb), yb); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        losses.append(tot / 4)
    np.testing.assert_allclose(h["loss"], losses, rtol=1e-5)
    layer = tr.model.layers[0]
    for v, p in zip(layer.vs, ref.parameters()):
        np.testing.assert_allclose(v.numpy(), p.detach().numpy(), atol=1e-5)  # identical weights after 12 steps


def test_torch_batchnorm_buffers_dropout_embedding():
    x, y = data(64)
    torch.manual_seed(1)
    mk = lambda: torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.BatchNorm1d(8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
    fm = mk(); ref = mk(); ref.load_state_dict(fm.state_dict())
    layer = an.Torch(fm)
    tr = an.Trainer(an.Sequential(layer), an.cross_entropy(), an.SGD(0.05))
    tr.fit(batches(x, y, 16), epochs=2, verbose=0)
    opt = torch.optim.SGD(ref.parameters(), lr=0.05)
    for _ in range(2):
        for i in range(0, 64, 16):
            xb, yb = torch.from_numpy(x[i:i + 16]), torch.from_numpy(y[i:i + 16]).long()
            loss = torch.nn.functional.cross_entropy(ref(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
    np.testing.assert_allclose(layer.bvs[0].numpy(), ref[1].running_mean.numpy(), atol=1e-5)   # running mean (buffer)
    np.testing.assert_allclose(layer.bvs[1].numpy(), ref[1].running_var.numpy(), atol=1e-5)    # running var
    ref.eval()                                                                                   # eval uses running stats
    np.testing.assert_allclose(tr.predict(x[:16], batch_size=16), ref(torch.from_numpy(x[:16])).detach().numpy(), atol=1e-4)
    # dropout: stochastic in training, deterministic in eval
    d = an.Torch(torch.nn.Dropout(0.5)); xx = np.ones((8, 20), "float32"); d(xx)
    a, b = d(xx, training=True).numpy(), d(xx, training=True).numpy()
    assert not np.array_equal(a, b) and np.array_equal(d(xx).numpy(), xx)
    # integer inputs (nn.Embedding): gradient reaches the table, none for the ids
    e = an.Torch(torch.nn.Embedding(10, 4)); ids = np.array([[1, 2], [2, 3]], "int32"); e(ids)
    with tf.GradientTape() as t: l = tf.reduce_sum(e(ids, training=True) ** 2)
    g = t.gradient(l, e.vs[0]).numpy(); assert g[2].any() and not g[0].any() and not g[9].any()


def test_torch_mixed_model_checkpoint_and_limits():
    x, y = data(64)
    torch.manual_seed(2)
    blk = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.GELU())
    mk = lambda: an.Trainer(an.Sequential(an.Dense(16, "relu"), an.Torch(blk), an.Dense(3)), an.cross_entropy(),
                            an.AdamW(1e-2), ema=0.9, precision="bfloat16", metrics=[an.Accuracy()])
    tr = mk(); h = tr.fit(x, y, epochs=12, batch_size=16, verbose=0)
    an.set_precision("float32")
    assert np.isfinite(h["loss"]).all() and h["loss"][-1] < 0.8 * h["loss"][0]
    assert len(tr.model.trainable_variables) == 6                       # dense w,b + torch w,b + dense w,b
    with tempfile.TemporaryDirectory() as d:
        p1 = tr.predict(x[:16], batch_size=16); tr.save(d + "/c")
        t2 = mk(); t2.load(d + "/c"); t2.init(x[:16])
        np.testing.assert_allclose(t2.predict(x[:16], batch_size=16), p1, atol=1e-5)    # torch params restored from the checkpoint
    an.set_precision("float32")
    class TwoOut(torch.nn.Module):
        def forward(self, z): return z, z
    try:
        an.Torch(TwoOut())(np.ones((2, 3), "float32")); raise AssertionError("multi-output must be rejected")
    except TypeError as e:
        assert "single output" in str(e)
    try:
        an.Sequential(an.Torch(torch.nn.Linear(3, 2))).get_config(); raise AssertionError
    except (ValueError, TypeError):
        pass  # not serialisable, and says so


# ===================================================================== JAX
def jmlp(p, x):
    return jnp.tanh(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


def jparams(seed=0):
    rs = np.random.RandomState(seed)
    return {"w1": rs.randn(5, 8).astype("float32") * .3, "b1": np.zeros(8, "float32"),
            "w2": rs.randn(8, 3).astype("float32") * .3, "b2": np.zeros(3, "float32")}


def test_jax_training_matches_native_jax():
    x, y = data(64)
    P = jparams()
    jl = an.Jax(jmlp, P)
    tr = an.Trainer(an.Sequential(jl), an.cross_entropy(), an.SGD(0.1, momentum=0.9))
    assert tr.static_batches                                         # switched on automatically for an.Jax layers
    h = tr.fit(batches(x, y, 16), epochs=3, verbose=0)
    # the very same training, natively in JAX
    def loss(p, xb, yb):
        lp = jax.nn.log_softmax(jmlp(p, xb)); return -jnp.mean(lp[jnp.arange(len(yb)), yb])
    gfn = jax.jit(jax.value_and_grad(loss)); p = jax.tree_util.tree_map(jnp.asarray, P)
    mom = jax.tree_util.tree_map(jnp.zeros_like, p); losses = []
    for _ in range(3):
        tot = 0.0
        for i in range(0, 64, 16):
            l, g = gfn(p, x[i:i + 16], y[i:i + 16])
            mom = jax.tree_util.tree_map(lambda m, gg: 0.9 * m + gg, mom, g)
            p = jax.tree_util.tree_map(lambda pp, m: pp - 0.1 * m, p, mom); tot += float(l)
        losses.append(tot / 4)
    np.testing.assert_allclose(h["loss"], losses, rtol=1e-5)
    got = jl.params()
    for k in P: np.testing.assert_allclose(got[k], np.asarray(p[k]), atol=1e-5)               # identical weights after 12 steps
    np.testing.assert_allclose(tr.predict(x[:16], batch_size=16), np.asarray(jmlp(got, x[:16])), atol=1e-5)


def test_jax_static_batches_ragged_data_and_dynamic_error():
    x, y = data(70)                                                   # 70 = 4*16 + 6: ragged tail
    jl = an.Jax(jmlp, jparams())
    tr = an.Trainer(an.Sequential(jl), an.cross_entropy(), an.SGD(0.05), metrics=[an.Accuracy()])
    tr.fit(x, y, epochs=1, batch_size=16, verbose=0)
    assert int(tr.opt.iterations.numpy()) == 4                        # ragged tail dropped, never a shape error
    pred = tr.predict(x, batch_size=16)
    assert pred.shape == (70, 3)
    np.testing.assert_allclose(pred, np.asarray(jmlp(jl.params(), x)), atol=1e-5)    # padded predict, exact rows
    ev = tr.evaluate(x, y, batch_size=16)
    assert abs(ev["acc"] - float((pred.argmax(1) == y).mean())) < 1e-9
    f = tf.function(lambda z: jl(z), input_signature=[tf.TensorSpec([None, 5], tf.float32)])
    try:
        f.get_concrete_function(); raise AssertionError("dynamic batch must be rejected")
    except ValueError as e:
        assert "static batch" in str(e)
    try:
        an.Jax(jmlp, {"a": np.zeros(3, "int32")}); raise AssertionError
    except TypeError:
        pass


def test_jax_rng_training_flag_and_mixed_models():
    x, y = data(64)
    def drop_mlp(p, x, key, training):
        h = jnp.tanh(x @ p["w1"] + p["b1"])
        if training:
            h = h * jax.random.bernoulli(key, 0.5, h.shape) * 2.0
        return h @ p["w2"] + p["b2"]
    jl = an.Jax(drop_mlp, jparams(), rng=True)
    xb = x[:16]; jl(xb)
    a, b = jl(xb, training=True).numpy(), jl(xb, training=True).numpy()
    assert not np.array_equal(a, b)                                  # fresh key each training call
    np.testing.assert_array_equal(jl(xb).numpy(), jl(xb).numpy())    # eval: deterministic, dropout off
    # ApexNet layers + JAX layer + Adam + EMA + bf16 (jax receives bf16 activations), then checkpoint
    jl2 = an.Jax(jmlp, jparams(1))
    mk = lambda: an.Trainer(an.Sequential(an.Dense(5, "gelu"), jl2), an.cross_entropy(), an.Adam(1e-2), ema=0.9,
                            precision="bfloat16", metrics=[an.Accuracy()])
    tr = mk(); h = tr.fit(x, y, epochs=15, batch_size=16, verbose=0); an.set_precision("float32")
    assert np.isfinite(h["loss"]).all() and h["loss"][-1] < 0.8 * h["loss"][0]
    with tempfile.TemporaryDirectory() as d:
        p1 = tr.predict(x[:16], batch_size=16); tr.save(d + "/c")
        t2 = mk(); t2.load(d + "/c"); t2.init(x[:16])
        np.testing.assert_allclose(t2.predict(x[:16], batch_size=16), p1, atol=1e-5)
    an.set_precision("float32")


def test_torch_and_jax_in_one_model_gradients_reach_both():
    x, y = data(64)
    torch.manual_seed(3)
    tl = an.Torch(torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.ReLU(), torch.nn.Linear(6, 3)))
    jl = an.Jax(jmlp, jparams(2))
    m = an.Sequential(jl, tl)                                          # JAX block -> PyTorch block
    tr = an.Trainer(m, an.cross_entropy(), an.Adam(2e-2), metrics=[an.Accuracy()])
    tr.init(x[:16])
    w0 = [v.numpy().copy() for v in m.trainable_variables]
    h = tr.fit(x, y, epochs=20, batch_size=16, verbose=0)
    moved = [not np.allclose(a, b.numpy()) for a, b in zip(w0, m.trainable_variables)]
    assert all(moved), moved                                           # every JAX and every torch parameter was updated
    assert h["loss"][-1] < 0.7 * h["loss"][0] and h["acc"][-1] > 0.7, (h["loss"][0], h["loss"][-1], h["acc"][-1])


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
