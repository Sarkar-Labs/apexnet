import os, sys, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import apexnet as an
H = an.hardware


def toy(n, seed=0, d=8, k=4):
    rs = np.random.RandomState(seed); x = rs.randn(n, d).astype("float32")
    return x, x[:, :k].argmax(1).astype("int32")


def test_recommendations_and_setup():
    r = H.recommend_precision
    assert r("tpu") == "bfloat16" and r("cpu") == "float32" and r("gpu") is not None
    assert r("gpu", (8, 0)) == "bfloat16" and r("gpu", (9, 0)) == "bfloat16"          # A100 / H100
    assert r("gpu", (7, 5)) == "float16" and r("gpu", (7, 0)) == "float16"            # T4 / V100
    assert r("gpu", (6, 0)) == "float32"                                              # P100 (Kaggle)
    assert H.recommend_steps_per_call("tpu") > H.recommend_steps_per_call("gpu") > H.recommend_steps_per_call("cpu") >= 1
    hw = H.setup(verbose=False)
    assert hw.kind in ("cpu", "gpu", "tpu") and hw.replicas >= 1 and hw.global_batch(8) == 8 * hw.replicas
    d = json.loads(json.dumps(hw.to_dict())); assert d["tensorflow"] == tf.__version__ and "strategy" in d
    assert set(hw.trainer_kwargs()) == {"strategy", "precision", "steps_per_call", "jit"}
    assert isinstance(an.auto_strategy(), tf.distribute.Strategy) and H.platform_name() in ("local", "colab", "kaggle")
    # TPU connection attempts must fail loudly and informatively (and never hang) on a machine without a TPU
    try:
        H.connect_tpu("local"); raise AssertionError("no TPU here")
    except RuntimeError as e:
        assert "tpu='local'" in str(e) and "Attempts" in str(e)
    assert H.checkpoint_options(tf.distribute.get_strategy()) is None


def test_auto_knobs_resolve_on_cpu():
    tr = an.Trainer(an.models.MLP([4], 2), an.cross_entropy(), an.Adam(), precision="auto", jit="auto", steps_per_call="auto")
    assert tr.kind in ("cpu", "gpu") and tr.build_mode == "eager" and not tr.static_batches and not tr.is_tpu
    if tr.kind == "cpu":
        assert tr.jit is False and tr.steps_per_call == H.recommend_steps_per_call("cpu")
    an.set_precision("float32")


def test_static_batches_exactness():
    """TPU-style static shapes, run on CPU: training drops the ragged tail; eval/predict padding is invisible."""
    x, y = toy(100)
    mk = lambda **kw: an.Trainer(an.Sequential(an.Dense(16, "tanh"), an.Dense(4)), an.cross_entropy(), an.SGD(0.05),
                                 metrics=[an.Accuracy(), an.MacroF1(4), an.AUC(256)] if False else [an.Accuracy(), an.MacroF1(4)], **kw)
    tf.random.set_seed(0)
    dyn, st = mk(), mk(static_batches=True)
    dyn.init(x[:32]); st.init(x[:32]); st.set_weights(dyn.get_weights())
    # evaluate (N=100, batch 32 -> last batch is 4 real + 28 padded examples)
    a, b = dyn.evaluate(x, y, batch_size=32), st.evaluate(x, y, batch_size=32)
    for k in a: np.testing.assert_allclose(a[k], b[k], rtol=1e-6, atol=1e-7, err_msg=k)
    # weighted 3-tuple datasets keep their weights; padded rows get weight 0
    w = np.random.RandomState(1).rand(100).astype("float32")
    ds3 = tf.data.Dataset.from_tensor_slices((x, y, w)).batch(32)
    a3, b3 = dyn.evaluate(ds3), st.evaluate(ds3)
    for k in a3: np.testing.assert_allclose(a3[k], b3[k], rtol=1e-6, atol=1e-7, err_msg=k)
    # predict: exactly N rows, same values, same order
    pa, pb = dyn.predict(x, batch_size=32), st.predict(x, batch_size=32)
    assert pb.shape == (100, 4); np.testing.assert_allclose(pa, pb, atol=1e-6)
    # training: 100/32 -> 3 full batches, ragged tail dropped, counters exact
    st.fit(x, y, epochs=1, batch_size=32, verbose=0)
    assert int(st.opt.iterations.numpy()) == 3 and float(st.train_meter.v["loss"][1].numpy()) == 96
    # a dataset that is already batched with a dynamic dim gets re-batched statically
    ds_dyn = tf.data.Dataset.from_tensor_slices((x, y)).batch(32)
    assert ds_dyn.element_spec[0].shape[0] is None
    t2 = mk(static_batches=True); t2.fit(ds_dyn, epochs=1, verbose=0)
    assert int(t2.opt.iterations.numpy()) == 3
    # variable-length examples cannot be padded to a static shape: clear error
    ragged = tf.data.Dataset.from_generator(lambda: ((np.ones(n, "float32"), 0) for n in (2, 3, 4, 5)),
                                            output_signature=(tf.TensorSpec([None], tf.float32), tf.TensorSpec([], tf.int32))).padded_batch(2)
    try:
        t2.evaluate(ragged); raise AssertionError("should refuse")
    except ValueError as e:
        assert "fixed per-example shapes" in str(e)


def test_trace_build_mode_equals_eager():
    x, _ = toy(8)
    mk = lambda mode: an.Trainer(an.Sequential(an.Dense(8, "gelu"), an.BatchNorm(), an.Dense(4)), an.cross_entropy(), an.Adam(), build_mode=mode)
    a, b = mk("eager"), mk("trace")
    a.init(x); b.init(x)
    assert b.build_info == "trace" and a.build_info == "eager"
    assert [v.shape for v in a.model.variables] == [v.shape for v in b.model.variables]
    b.set_weights(a.get_weights()); np.testing.assert_allclose(a.predict(x), b.predict(x), atol=1e-6)
    # recurrent + attention models also build by tracing
    for m, xx in [(an.Sequential(an.LSTM(5), an.Dense(3)), np.random.randn(4, 6, 2).astype("float32")),
                  (an.models.TransformerLM(11, 16, 1, 2, 8), np.random.randint(0, 11, (4, 8)).astype("int32"))]:
        t = an.Trainer(m, an.cross_entropy(), an.Adam(), build_mode="trace"); t.init(xx)
        assert t.build_info == "trace" and len(m.variables) > 0 and np.isfinite(t.predict(xx)).all()


def test_exact_step_counts_and_wraparound():
    x, y = toy(160)
    mk = lambda spc: an.Trainer(an.Sequential(an.Dense(4)), an.cross_entropy(), an.SGD(0.01), steps_per_call=spc)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(32, drop_remainder=True)       # 5 batches, known cardinality
    for spc in (1, 2, 3, 7):
        t = mk(spc); t.fit(ds, epochs=2, verbose=0)
        assert int(t.opt.iterations.numpy()) == 10, spc
    t = mk(3); t.fit(ds, epochs=2, steps_per_epoch=8, verbose=0)                        # finite data, 8 steps/epoch: wraps
    assert int(t.opt.iterations.numpy()) == 16 and float(t.train_meter.v["loss"][1].numpy()) == 8 * 32
    unk = tf.data.Dataset.from_tensor_slices((x, y)).filter(lambda a, b: True).batch(32, drop_remainder=True)
    assert int(unk.cardinality()) == tf.data.UNKNOWN_CARDINALITY
    t = mk(2); t.fit(unk, epochs=2, verbose=0); assert int(t.opt.iterations.numpy()) == 10  # OutOfRange fallback
    inf = ds.repeat(); t = mk(4); t.fit(inf, epochs=2, steps_per_epoch=6, verbose=0); assert int(t.opt.iterations.numpy()) == 12


def test_optimizer_state_is_lazy_and_checkpointed():
    o = an.AdamW(1e-3)
    assert o._iterations is None                       # nothing created outside a strategy scope
    assert int(o.iterations.numpy()) == 0 and float(o.lr_scale.numpy()) == 1.0
    la = an.Lookahead(an.SGD(0.1)); la._state(); assert la.iterations is la.inner.iterations
    x, y = toy(64)
    with tempfile.TemporaryDirectory() as d:
        t = an.Trainer(an.Sequential(an.Dense(4)), an.cross_entropy(), an.Adam(1e-2)); t.fit(x, y, epochs=2, batch_size=16, verbose=0)
        t.opt.lr_scale.assign(0.25); t.save(d + "/c")
        t2 = an.Trainer(an.Sequential(an.Dense(4)), an.cross_entropy(), an.Adam(1e-2)); t2.load(d + "/c"); t2.init(x[:2])
        assert int(t2.opt.iterations.numpy()) == 8 and abs(float(t2.opt.lr_scale.numpy()) - 0.25) < 1e-9


def test_selftest_runs_and_reports():
    with tempfile.TemporaryDirectory() as d:
        rep = an.selftest.run(quick=True, report_path=d + "/r.json", verbose=False)
        assert rep["summary"]["FAIL"] == 0, [c for c in rep["checks"] if c["status"] == "FAIL"]
        assert json.load(open(d + "/r.json"))["hardware"]["kind"] in ("cpu", "gpu", "tpu")
        assert {c["status"] for c in rep["checks"]} <= {"PASS", "SKIP"}
        bench = an.selftest.benchmark(H.setup(verbose=False), which=("mlp",), verbose=False)   # ApexNet-vs-Keras harness works
        assert bench[0]["apexnet"] > 0 and bench[0].get("keras_default", 1) > 0 and "apexnet_error" not in bench[0]


def test_each_trainer_uses_its_own_precision():
    """The precision policy is process-global state; a Trainer built later must not change another Trainer's dtype."""
    x, _ = toy(32)
    mk = lambda prec: an.Trainer(an.models.MLP([32], 4, "gelu"), an.cross_entropy(), an.Adam(), precision=prec)
    a = mk("bfloat16"); b = mk("float32")        # b is constructed LAST: the global policy is now float32
    a.init(x[:16]); b.init(x[:16]); b.set_weights(a.get_weights())
    pa, pb = a.predict(x, batch_size=16), b.predict(x, batch_size=16)
    assert not np.allclose(pa, pb, atol=1e-6), "trainer `a` silently computed in float32"
    assert np.abs(pa - pb).max() < 0.15           # bf16 vs fp32: same function, bf16-level difference
    np.testing.assert_array_equal(pa, a.predict(x, batch_size=16))                   # stable regardless of call order
    from apexnet.core import compute_dtype
    assert compute_dtype() == tf.float32          # global policy restored after the call
    an.set_precision("float32")


def test_xla_fallback_for_uncompilable_models():
    """XLA is on by default for a single GPU but never on CPU, so 'works on CPU' did not imply 'works on GPU'.
    A model XLA cannot compile must fall back automatically (it used to raise, or abort the interpreter, on GPU)."""
    texts = np.array(["red green", "blue red red", "green"] * 22)[:64]; yt = np.tile([0, 1, 2], 22)[:64].astype("int32")
    tv = an.TextVectorizer(max_len=4).adapt(texts)                       # string ops cannot be XLA-compiled
    tr = an.Trainer(an.Sequential(tv, an.Embedding(tv.vocab_size, 4), an.GlobalAvgPool(), an.Dense(3)),
                    an.cross_entropy(), an.Adam(1e-2), jit=True)
    h = tr.fit(texts, yt, epochs=2, batch_size=16, verbose=0)
    assert np.isfinite(h["loss"]).all() and tr.jit is False and tr.jit_note.startswith("XLA disabled")
    assert tr.evaluate(texts, yt, batch_size=16)["loss"] > 0
    x, y = toy(64)                                                        # a compilable model keeps XLA
    ok = an.Trainer(an.models.MLP([8], 4), an.cross_entropy(), an.Adam(), jit=True)
    ok.fit(x, y, epochs=1, batch_size=16, verbose=0)
    assert ok.jit is True and ok.jit_note is None and ok._xla_checked["train"]
    ok.evaluate(x, y, batch_size=16); assert ok._xla_checked["eval"]
    assert set(ok.settings()) >= {"device", "precision", "xla_jit", "steps_per_call", "strategy"}
    try:
        import torch
        t2 = an.Trainer(an.Sequential(an.Dense(8, "relu"), an.Torch(torch.nn.Linear(8, 4))), an.cross_entropy(), an.Adam(), jit=True)
        t2.fit(x, y, epochs=1, batch_size=16, verbose=0); assert t2.jit is False   # tf.py_function: not XLA-compilable
    except ImportError:
        pass


def test_simulated_t4_defaults_train_on_cpu():
    """Pretend a Tesla T4 is attached: every 'auto' resolves to fp16 + XLA + steps_per_call=8 (what a Colab GPU runtime
    gets). Run several model families through exactly that configuration on the CPU."""
    real = H.gpu_info
    H.gpu_info = lambda: [{"name": "Tesla T4 (simulated)", "compute_capability": (7, 5)}]
    try:
        rs = np.random.RandomState(0); t = rs.randint(0, 16, (32, 8)).astype("int32"); yv = rs.randint(0, 4, 32).astype("int32")
        img = rs.randn(32, 8, 8, 3).astype("float32")
        for name, m, xx, yy in [("resnet", an.models.ResNet(4, (1, 1), (4, 8)), img, yv),
                                ("lstm", an.Sequential(an.LSTM(5), an.Dense(4)), rs.randn(32, 6, 3).astype("float32"), yv),
                                ("transformer", an.models.TransformerLM(16, 16, 1, 2, 8), t, np.roll(t, -1, 1))]:
            tr = an.Trainer(m, an.cross_entropy(), an.AdamW(3e-3), precision="auto", jit="auto", steps_per_call="auto", ema=0.9)
            assert tr.kind == "gpu" and tr.precision_name == "float16" and tr.steps_per_call == 8
            h = tr.fit(xx, yy, epochs=2, batch_size=16, verbose=0)
            assert np.isfinite(h["loss"]).all() and np.isfinite(tr.predict(xx, batch_size=16)).all(), name
            assert tr.scaler is not None and tr.jit is True, name           # fp16 loss scaling + XLA really were active
    finally:
        H.gpu_info = real
        an.set_precision("float32")


def test_setup_configures_memory_growth_before_touching_gpus():
    """Querying GPU details can initialise them, after which memory growth can no longer be set (TF then grabs all VRAM)."""
    calls = []
    class FakeGPU:
        name = "/physical_device:GPU:0"
    real = (tf.config.list_physical_devices, tf.config.experimental.set_memory_growth, H.gpu_info)
    tf.config.list_physical_devices = lambda kind=None: [FakeGPU()] if kind == "GPU" else real[0](kind)
    tf.config.experimental.set_memory_growth = lambda g, v: calls.append("set_memory_growth")
    H.gpu_info = lambda: (calls.append("gpu_info"), [{"name": "fake", "compute_capability": (8, 0)}])[1]
    try:
        hw = H.setup(prefer="gpu", verbose=False)
    finally:
        tf.config.list_physical_devices, tf.config.experimental.set_memory_growth, H.gpu_info = real
    assert calls.index("set_memory_growth") < calls.index("gpu_info"), calls
    assert hw.kind == "gpu" and hw.precision == "bfloat16"                  # Ampere -> bf16


def test_counters_are_int64_not_int32():
    x, y = toy(32)
    tr = an.Trainer(an.models.MLP([8], 4), an.cross_entropy(), an.Adam(), accumulate=2, precision="float16")
    tr.fit(x, y, epochs=1, batch_size=16, verbose=0)
    assert tr.micro.dtype == tr.skipped.dtype == tr.scaler.good.dtype == tf.int64      # int32 variables are a GPU hazard
    an.set_precision("float32")


def _os(system="linux", machine="x86_64", **kw):
    d = {"system": system, "release": "x", "machine": machine, "arm": machine in ("arm64", "aarch64"),
         "apple_silicon": system == "macos" and machine == "arm64", "wsl": False, "container": False, "cpu_cores": 8,
         "python": "3.12", "hosted": "local", "nvidia_smi": False, "rocm_smi": False}
    d.update(kw)
    return d


def test_choose_hardware_and_os_scenarios():
    T4 = {"name": "T4", "compute_capability": (7, 5)}
    ch = lambda dev, osi, prefer="auto", env=None: H.choose(dev, osi, prefer, env or {})
    # priority: TPU > GPU > CPU
    assert ch({"gpu": [T4], "tpu": False}, _os()).order == ["gpu", "cpu"]
    assert ch({"gpu": [T4], "tpu": True}, _os()).order == ["tpu", "gpu", "cpu"]
    assert ch({"gpu": [], "tpu": False}, _os()).order == ["cpu"] and ch({"gpu": [T4] * 2}, _os()).kind == "gpu"
    # explicit choice is strict (CPU is the last resort); the env var beats the argument
    assert ch({"gpu": [T4]}, _os(), "cpu").order == ["cpu"] and ch({"gpu": [T4]}, _os(), "gpu").order == ["gpu", "cpu"]
    assert ch({"gpu": [T4]}, _os(), "auto", {"APEXNET_DEVICE": "cpu"}).order == ["cpu"]
    assert "env APEXNET_DEVICE" in " ".join(ch({"gpu": [T4]}, _os(), "auto", {"APEXNET_DEVICE": "cpu"}).reasons)
    try:
        ch({"gpu": []}, _os(), "quantum"); raise AssertionError
    except ValueError as e:
        assert "auto|cpu|gpu|tpu" in str(e)
    # OS-specific guidance
    w = ch({"gpu": []}, _os("windows", nvidia_smi=True)); assert "WSL2" in " ".join(w.notes) and "NVIDIA GPU is present" in " ".join(w.notes)
    assert "WSL2" in " ".join(ch({"gpu": []}, _os("windows")).notes)
    m = ch({"gpu": []}, _os("macos", "arm64")); assert "tensorflow-metal" in " ".join(m.notes) and m.kind == "cpu"
    mg = ch({"gpu": [{"name": "Apple M2", "compute_capability": None}]}, _os("macos", "arm64"))
    assert mg.kind == "gpu" and mg.jit_ok is False and "Metal" in " ".join(mg.notes)      # Apple GPU: no XLA
    assert ch({"gpu": [T4]}, _os()).jit_ok is True
    l = ch({"gpu": []}, _os("linux", nvidia_smi=True)); assert "tensorflow[and-cuda]" in " ".join(l.notes)
    assert "tensorflow-rocm" in " ".join(ch({"gpu": []}, _os("linux", rocm_smi=True)).notes)
    assert any("WSL2" in n for n in ch({"gpu": [T4]}, _os("linux", wsl=True)).notes)
    assert any("4 CPU core" in n for n in ch({"gpu": []}, _os("linux", container=True, cpu_cores=4)).notes)
    # precision rules per OS / GPU generation
    r = H.recommend_precision
    assert r("gpu", (8, 0), "linux") == "bfloat16" and r("gpu", (7, 5), "windows") == "float16" and r("gpu", (8, 0), "macos") == "float32"
    assert r("gpu", None, "macos") == "float32" and r("tpu", None, "macos") == "bfloat16"


def test_os_info_and_describe_and_auto_api():
    import platform
    real = (platform.system, platform.machine, platform.release)
    try:
        platform.system, platform.machine, platform.release = (lambda: "Darwin"), (lambda: "arm64"), (lambda: "23.1.0")
        o = H.os_info(); assert o["system"] == "macos" and o["apple_silicon"] and o["arm"] and not o["wsl"]
        platform.system, platform.machine, platform.release = (lambda: "Windows"), (lambda: "AMD64"), (lambda: "10")
        o = H.os_info(); assert o["system"] == "windows" and not o["apple_silicon"]
        platform.system, platform.machine, platform.release = (lambda: "Linux"), (lambda: "x86_64"), (lambda: "5.15.90.1-microsoft-standard-WSL2")
        assert H.os_info()["wsl"] is True
    finally:
        platform.system, platform.machine, platform.release = real
    o = H.os_info(); assert o["cpu_cores"] >= 1 and o["system"] in ("linux", "windows", "macos")
    hw = H.setup(verbose=False)
    box = H.describe(hw)
    assert "ApexNet hardware" in box and hw.os["system"] in box and "selected" in box and hw.kind.upper() in box
    assert hw.to_dict()["os_info"]["system"] == hw.os["system"] and isinstance(hw.to_dict()["why"], list)
    # env overrides (APEXNET_DEVICE / APEXNET_PRECISION)
    os.environ["APEXNET_DEVICE"], os.environ["APEXNET_PRECISION"] = "cpu", "bfloat16"
    try:
        h2 = an.auto(verbose=False); assert h2.kind == "cpu" and h2.precision == "bfloat16"
    finally:
        del os.environ["APEXNET_DEVICE"], os.environ["APEXNET_PRECISION"]
    # one-call trainer: Trainer.auto / strategy="auto" / strategy=<Hardware>
    x, y = toy(32)
    t1 = an.Trainer.auto(an.models.MLP([8], 4), an.cross_entropy(), an.Adam(1e-2))
    assert isinstance(t1.hardware, H.Hardware) and t1.kind == t1.hardware.kind
    t1.fit(x, y, epochs=1, batch_size=16, verbose=0)
    t2 = an.Trainer(an.models.MLP([8], 4), an.cross_entropy(), an.Adam(), strategy="auto"); t2.fit(x, y, epochs=1, batch_size=16, verbose=0)
    t3 = an.Trainer(an.models.MLP([8], 4), an.cross_entropy(), an.Adam(), strategy=hw); assert t3.hardware is hw
    try:
        an.Trainer(an.models.MLP([8], 4), an.cross_entropy(), an.Adam(), strategy="tpu"); raise AssertionError
    except ValueError:
        pass
    an.set_precision("float32")


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
