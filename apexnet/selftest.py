"""Hardware self-test + ApexNet-vs-Keras benchmark. Run it on the machine you care about:

    import apexnet as an
    an.selftest.run()                  # all checks + benchmark, writes apexnet_hw_report.json
    an.selftest.run(quick=True)        # checks only, ~1 minute
    # or:  python -m apexnet.selftest [--quick] [--prefer cpu|gpu|tpu]

Checks: accelerator vs CPU reference (fp32), exact metrics under static-shape padding, mixed precision, XLA,
multi-step loops, model smoke tests (LSTM / Transformer / BatchNorm CNN), checkpoint + SavedModel + save_model,
then a throughput benchmark of ApexNet vs Keras under the SAME tf.distribute strategy and precision.
"""
import argparse
import json
import os
import sys
import time
import traceback
import numpy as np
import tensorflow as tf

from . import hardware as H

TOL = {"cpu": 1e-5, "gpu": 3e-4, "tpu": 6e-2}   # relative tolerances vs the CPU reference (TPU matmuls use bf16 passes)


class Skip(Exception):
    pass


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-8))


def _an():
    import apexnet as an
    return an


def _toy(n, d=16, k=4, seed=0):
    rs = np.random.RandomState(seed)
    x = rs.randn(n, d).astype("float32")
    y = (x[:, :k].argmax(1)).astype("int32")
    return x, y


# ===================================================================== checks
def check_equivalence(hw):
    an = _an()
    gb = hw.global_batch(16)
    x, y = _toy(gb * 8)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(gb, drop_remainder=True)
    mk = lambda: an.Sequential(an.Dense(32, "tanh"), an.LayerNorm(), an.Dense(4))
    tf.random.set_seed(0)
    with tf.device("/CPU:0"):
        ref = an.Trainer(mk(), an.cross_entropy(), an.SGD(0.1, momentum=0.9), metrics=[an.Accuracy()],
                         strategy=tf.distribute.get_strategy(), precision="float32", jit=False)
        ref.init(x[:gb]); w0 = ref.get_weights()
        h_ref = ref.fit(ds, epochs=2, verbose=0)
    acc = an.Trainer(mk(), an.cross_entropy(), an.SGD(0.1, momentum=0.9), metrics=[an.Accuracy()],
                     strategy=hw.strategy, precision="float32", jit=False)
    acc.init(x[:gb]); acc.set_weights(w0)
    h = acc.fit(ds, epochs=2, verbose=0)
    tol = TOL[hw.kind]
    dl, dw = _rel(h["loss"], h_ref["loss"]), max(_rel(a, b) for a, b in zip(acc.get_weights(), ref.get_weights()))
    assert dl < tol and dw < tol * 10, f"loss rel.diff {dl:.2e}, weight rel.diff {dw:.2e} (tol {tol:.0e})"
    return f"loss rel.diff {dl:.1e}, weights rel.diff {dw:.1e} (tol {tol:.0e})"


def check_padding_exactness(hw):
    """Odd-sized eval set, static (padded) batches: evaluate() must equal numpy computed from predict()."""
    an = _an()
    gb = hw.global_batch(8)
    n = gb * 5 + 3                                  # never a multiple of the batch
    x, y = _toy(n)
    tr = an.Trainer(an.Sequential(an.Dense(16, "relu"), an.Dense(4)), an.cross_entropy(), an.SGD(0.0),
                    metrics=[an.Accuracy(), an.MacroF1(4)], strategy=hw.strategy, precision="float32",
                    static_batches=True)
    tr.init(x[:gb])
    pred = tr.predict(x, batch_size=gb)
    assert pred.shape == (n, 4), pred.shape
    ev = tr.evaluate(x, y, batch_size=gb)
    ex = np.exp(pred - pred.max(1, keepdims=True)); lp = np.log(ex / ex.sum(1, keepdims=True))
    ref_loss = float(-lp[np.arange(n), y].mean())
    ref_acc = float((pred.argmax(1) == y).mean())
    assert abs(ev["loss"] - ref_loss) < 2e-4 * max(1, ref_loss), (ev["loss"], ref_loss)
    assert abs(ev["acc"] - ref_acc) < 1e-9, (ev["acc"], ref_acc)
    return f"n={n} (batch {gb}): loss {ev['loss']:.5f}=={ref_loss:.5f}, acc {ev['acc']:.4f}"


def check_trace_build(hw):
    """TPU path: variables created by tracing (no host forward pass). The Trainer falls back to an eager build if this
    fails, so a FAIL here is informational."""
    if hw.kind != "tpu":
        raise Skip("only relevant on TPU")
    an = _an()
    gb = hw.global_batch(8)
    x, y = _toy(gb * 2)
    tr = an.Trainer(an.models.MLP([16], 4), an.cross_entropy(), an.Adam(1e-2), strategy=hw.strategy, build_mode="trace")
    h = tr.fit(x, y, epochs=1, batch_size=gb, verbose=0)
    assert np.isfinite(h["loss"]).all()
    return f"build path used: {tr.build_info}"


def check_mixed_precision(hw):
    an = _an()
    prec = hw.precision if hw.precision != "float32" else "bfloat16"
    gb = hw.global_batch(32)
    x, y = _toy(gb * 8)
    tr = an.Trainer(an.models.MLP([64, 64], 4, "gelu"), an.cross_entropy(), an.AdamW(3e-3), metrics=[an.Accuracy()],
                    strategy=hw.strategy, precision=prec, steps_per_call=hw.steps_per_call,
                    static_batches=hw.kind == "tpu")
    h = tr.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(gb, drop_remainder=True).repeat(), epochs=6,
               steps_per_epoch=8, verbose=0)
    assert np.isfinite(h["loss"]).all() and h["loss"][-1] < h["loss"][0], h["loss"]
    assert tr.predict(x[:gb]).dtype == np.float32
    extra = f", loss scale {float(tr.scaler.scale.numpy()):g}" if tr.scaler is not None else ""
    return f"{prec}: loss {h['loss'][0]:.3f} -> {h['loss'][-1]:.3f}{extra}"


def check_xla(hw):
    if hw.kind == "tpu":
        raise Skip("TPU programs are always XLA-compiled")
    if hw.replicas > 1:
        raise Skip("XLA inside multi-replica collectives is not enabled by default")
    an = _an()
    x, y = _toy(256)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(64)
    mk = lambda: an.Sequential(an.Dense(32, "gelu"), an.Dense(4))
    tf.random.set_seed(0)
    a = an.Trainer(mk(), an.cross_entropy(), an.SGD(0.1), jit=False, precision="float32"); a.init(x[:64]); w0 = a.get_weights()
    b = an.Trainer(mk(), an.cross_entropy(), an.SGD(0.1), jit=True, precision="float32"); b.init(x[:64]); b.set_weights(w0)
    la, lb = a.fit(ds, epochs=2, verbose=0)["loss"], b.fit(ds, epochs=2, verbose=0)["loss"]
    assert _rel(lb, la) < max(TOL[hw.kind] * 10, 1e-3), (la, lb)
    return f"jit loss == eager loss (rel {_rel(lb, la):.1e})"


def check_xla_fallback(hw):
    """A model XLA cannot compile (string ops) must train anyway: Trainer(jit=True) falls back instead of crashing.
    On a GPU runtime this exercises the real XLA-GPU compiler."""
    if hw.kind == "tpu":
        raise Skip("TPU programs are always XLA-compiled")
    an = _an()
    texts = np.array(["red green", "blue red red", "green blue"] * 16)
    y = np.tile([0, 1, 2], 16).astype("int32")
    tv = an.TextVectorizer(max_len=4).adapt(texts)
    gb = hw.global_batch(8)
    tr = an.Trainer(an.Sequential(tv, an.Embedding(tv.vocab_size, 4), an.GlobalAvgPool(), an.Dense(3)), an.cross_entropy(),
                    an.Adam(1e-2), strategy=hw.strategy, precision="float32", jit=True if hw.replicas == 1 else False)
    h = tr.fit(texts, y, epochs=2, batch_size=gb, verbose=0)
    assert np.isfinite(h["loss"]).all()
    return f"trained; xla_jit={tr.jit}" + (f" ({tr.jit_note[:70]})" if tr.jit_note else "")


def check_multi_step(hw):
    an = _an()
    gb = hw.global_batch(16)
    spc = max(2, hw.steps_per_call)
    steps = 2 * spc + 3
    x, y = _toy(gb * steps)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(gb, drop_remainder=True)
    mk = lambda: an.Sequential(an.Dense(16, "tanh"), an.Dense(4))
    tf.random.set_seed(0)
    t1 = an.Trainer(mk(), an.cross_entropy(), an.SGD(0.05, 0.9), strategy=hw.strategy, precision="float32", steps_per_call=1, jit=False)
    t1.init(x[:gb]); w0 = t1.get_weights()
    t2 = an.Trainer(mk(), an.cross_entropy(), an.SGD(0.05, 0.9), strategy=hw.strategy, precision="float32", steps_per_call=spc, jit=False)
    t2.init(x[:gb]); t2.set_weights(w0)
    t1.fit(ds, epochs=1, verbose=0); t2.fit(ds, epochs=1, verbose=0)
    d = max(_rel(a, b) for a, b in zip(t2.get_weights(), t1.get_weights()))
    assert d < TOL[hw.kind] * 10, d
    assert int(t2.opt.iterations.numpy()) == steps == int(t1.opt.iterations.numpy())
    return f"{steps} steps, steps_per_call={spc}: weights rel.diff {d:.1e}"


def check_models(hw):
    an = _an()
    prec = hw.precision
    rs = np.random.RandomState(0)
    gb = hw.global_batch(8)
    specs = {
        "lstm": (an.Sequential(an.LSTM(16), an.Dense(3)), rs.randn(gb * 4, 12, 5).astype("float32"), rs.randint(0, 3, gb * 4).astype("int32")),
        "transformer": (an.models.TransformerLM(32, 32, 2, 2, 16), rs.randint(0, 32, (gb * 4, 16)).astype("int32"), rs.randint(0, 32, (gb * 4, 16)).astype("int32")),
        "cnn+bn": (an.Sequential(an.Conv2D(8, 3), an.BatchNorm(), an.Activation("relu"), an.MaxPool2D(2), an.GlobalAvgPool(), an.Dense(3)),
                   rs.randn(gb * 4, 12, 12, 3).astype("float32"), rs.randint(0, 3, gb * 4).astype("int32")),
    }
    out = []
    for name, (m, x, y) in specs.items():
        tr = an.Trainer(m, an.cross_entropy(), an.AdamW(1e-3), strategy=hw.strategy, precision=prec,
                        static_batches=hw.kind == "tpu")
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(gb, drop_remainder=True)
        h = tr.fit(ds, epochs=2, verbose=0)
        assert np.isfinite(h["loss"]).all(), name
        out.append(f"{name} {h['loss'][0]:.2f}->{h['loss'][-1]:.2f}")
    return "; ".join(out)


def check_interop(hw):
    """an.Torch / an.Jax layers: forward values and gradients must equal the native frameworks'."""
    an = _an()
    done = []
    rs = np.random.RandomState(0)
    x = rs.randn(8, 5).astype("float32")
    try:
        import jax, jax.numpy as jnp
        f = lambda p, z: jnp.tanh(z @ p["w"] + p["b"])
        P = {"w": rs.randn(5, 3).astype("float32"), "b": np.zeros(3, "float32")}
        layer = an.Jax(f, P)
        with tf.GradientTape() as t:
            y = layer(x, training=True)
            l = tf.reduce_sum(y ** 2)
        g = t.gradient(l, layer.vs)
        gref = jax.tree_util.tree_flatten(jax.grad(lambda p: jnp.sum(f(p, x) ** 2))(P))[0]
        assert _rel(y.numpy(), np.asarray(f(P, x))) < 1e-4 and all(_rel(a.numpy(), np.asarray(b)) < 1e-3 for a, b in zip(g, gref))
        done.append("jax")
    except ImportError:
        pass
    if hw.kind != "tpu":  # torch layers round-trip through the host
        try:
            import torch
            tm = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Tanh())
            ref = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Tanh()); ref.load_state_dict(tm.state_dict())
            layer = an.Torch(tm)
            with tf.GradientTape() as t:
                y = layer(x, training=True)
                l = tf.reduce_sum(y ** 2)
            g = t.gradient(l, layer.vs)
            (ref(torch.from_numpy(x)) ** 2).sum().backward()
            assert all(_rel(a.numpy(), p.grad.numpy()) < 1e-4 for a, p in zip(g, ref.parameters()))
            done.append("torch")
        except ImportError:
            pass
    if not done:
        raise Skip("neither jax nor torch is installed")
    return "gradients equal native: " + ", ".join(done)


def check_checkpoint_export(hw):
    import tempfile
    an = _an()
    gb = hw.global_batch(8)
    x, y = _toy(gb * 4)
    mk = lambda: an.Trainer(an.models.MLP([16], 4), an.cross_entropy(), an.Adam(1e-2), ema=0.9,
                            strategy=hw.strategy, precision="float32", static_batches=hw.kind == "tpu")
    with tempfile.TemporaryDirectory() as d:
        a = mk(); a.fit(x, y, epochs=2, batch_size=gb, verbose=0)
        p1 = a.predict(x[:gb])
        a.save(d + "/ck")
        b = mk(); b.load(d + "/ck"); b.init(x[:gb])
        assert _rel(b.predict(x[:gb]), p1) < 1e-5, "checkpoint restore mismatch"
        assert b.epoch == 2
        b.fit(x, y, epochs=3, batch_size=gb, verbose=0)                              # resumes (1 more epoch)
        a.export(d + "/sm", [tf.TensorSpec([None, 16], tf.float32)])
        with tf.device("/CPU:0"):
            loaded = tf.saved_model.load(d + "/sm")  # keep a reference: the object owns the variables
            served = loaded.serve(tf.constant(x[:gb])).numpy()
        assert _rel(served, p1) < max(TOL[hw.kind] * 10, 1e-4), _rel(served, p1)     # export uses EMA weights == predict
        an.save_model(a.model, d + "/mdl", x[:gb], options=H.checkpoint_options(hw.strategy))
        m2 = an.load_model(d + "/mdl")
        with a._maybe_ema(False):
            ref = a.predict(x[:gb], use_ema=False)
        assert _rel(m2(x[:gb]).numpy(), ref) < max(TOL[hw.kind] * 10, 1e-4)
    return "trainer save/load/resume, SavedModel export, save_model/load_model all consistent"


# ==================================================================== benchmark
def _cfg(kind):
    if kind == "cpu":
        return dict(S=20, R=4, mlp=(256, [512, 512], 10, 64), cnn=(32, [16, 32], 10, 16), tfm=(256, 64, 64, 2, 4, 8))
    return dict(S=50, R=4, mlp=(1024, [2048, 2048], 1000, 512), cnn=(64, [64, 128, 256], 100, 128),
                tfm=(8192, 256, 512, 4, 8, 32))


def _data(which, c, gb):
    rs = np.random.RandomState(0)
    if which == "mlp":
        d, _, k, _ = c["mlp"]; x, y = rs.randn(gb, d).astype("float32"), rs.randint(0, k, gb).astype("int32")
    elif which == "cnn":
        s, _, k, _ = c["cnn"]; x, y = rs.randn(gb, s, s, 3).astype("float32"), rs.randint(0, k, gb).astype("int32")
    else:
        V, T, *_ = c["tfm"]; x, y = rs.randint(0, V, (gb, T)).astype("int32"), rs.randint(0, V, (gb, T)).astype("int32")
    return tf.data.Dataset.from_tensors((x, y)).repeat()


def _apexnet_model(which, c):
    an = _an()
    if which == "mlp":
        _, hid, k, _ = c["mlp"]; return an.Sequential(*[an.Dense(h, "relu") for h in hid], an.Dense(k))
    if which == "cnn":
        _, ch, k, _ = c["cnn"]
        layers = []
        for f in ch:
            layers += [an.Conv2D(f, 3, activation="relu"), an.MaxPool2D(2)]
        return an.Sequential(*layers, an.GlobalAvgPool(), an.Dense(k))
    V, T, dim, depth, heads, _ = c["tfm"]
    return an.models.TransformerLM(V, dim, depth, heads, T)


def _keras_model(which, c, keras):
    L = keras.layers
    if which == "mlp":
        d, hid, k, _ = c["mlp"]
        return keras.Sequential([L.Input((d,))] + [L.Dense(h, activation="relu") for h in hid] + [L.Dense(k, dtype="float32")])
    if which == "cnn":
        s, ch, k, _ = c["cnn"]
        layers = [L.Input((s, s, 3))]
        for f in ch:
            layers += [L.Conv2D(f, 3, padding="same", activation="relu"), L.MaxPool2D(2)]
        return keras.Sequential(layers + [L.GlobalAveragePooling2D(), L.Dense(k, dtype="float32")])
    V, T, dim, depth, heads, _ = c["tfm"]
    inp = keras.Input((T,), dtype="int32")
    h = L.Embedding(V, dim)(inp)
    h = h + L.Embedding(T, dim)(keras.ops.arange(T)[None, :])
    for _ in range(depth):
        x1 = L.LayerNormalization()(h)
        h = h + L.MultiHeadAttention(heads, dim // heads)(x1, x1, use_causal_mask=True)
        h = h + L.Dense(dim)(L.Dense(4 * dim, activation="gelu")(L.LayerNormalization()(h)))
    return keras.Model(inp, L.Dense(V, dtype="float32")(L.LayerNormalization()(h)))


def _time_epochs(run_epochs, S, gb, R):
    """Median samples/s over R separately-timed epochs after one warm-up epoch (tracing / XLA compile excluded)."""
    run_epochs(1)
    times = []
    for _ in range(R):
        t0 = time.perf_counter()
        run_epochs(1)
        times.append(time.perf_counter() - t0)
    return gb * S / float(np.median(times))


def benchmark(hw, which=("mlp", "cnn", "tfm"), precision=None, verbose=True, baselines="auto"):
    import keras
    an = _an()
    c = _cfg(hw.kind)
    precision = precision or hw.precision
    kp = {"float32": "float32", "float16": "mixed_float16", "bfloat16": "mixed_bfloat16"}[precision]
    rows = []
    for w in which:
        gb = hw.global_batch(c[w][-1])
        ds = _data(w, c, gb)
        S, R = c["S"], c["R"]
        row = {"model": w, "precision": precision, "global_batch": gb}
        # ---- ApexNet
        try:
            tr = an.Trainer(_apexnet_model(w, c), an.cross_entropy(), an.Adam(1e-3), strategy=hw.strategy,
                            precision=precision, steps_per_call="auto", jit="auto", static_batches=hw.kind == "tpu")
            row["apexnet"] = _time_epochs(lambda r: tr.fit(ds, epochs=tr.epoch + r, steps_per_epoch=S, verbose=0), S, gb, R)
            row["apexnet_cfg"] = f"steps_per_call={tr.steps_per_call}, jit={tr.jit}"
        except Exception as e:  # noqa
            row["apexnet_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        # ---- Keras (out-of-the-box defaults, then tuned the same way ApexNet's "auto" is)
        keras.config.set_dtype_policy(kp)
        for label, spe, jit in (("keras_default", 1, "auto"), ("keras_tuned", hw.steps_per_call, hw.kind == "gpu" and hw.replicas == 1)):
            try:
                with hw.strategy.scope():
                    m = _keras_model(w, c, keras)
                    m.compile(optimizer=keras.optimizers.Adam(1e-3),
                              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              steps_per_execution=spe, jit_compile=jit)
                ep = {"n": 0}
                def run(r, m=m, ep=ep):
                    m.fit(ds, epochs=ep["n"] + r, initial_epoch=ep["n"], steps_per_epoch=S, verbose=0); ep["n"] += r
                row[label] = _time_epochs(run, S, gb, R)
            except Exception as e:  # noqa
                row[label + "_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            keras.backend.clear_session()
        keras.config.set_dtype_policy("float32")
        # ---- PyTorch / JAX baselines. Not on TPU (one process owns the TPU), and only if installed.
        fws = ("torch", "jax") if baselines == "auto" and hw.kind != "tpu" else (() if baselines == "auto" else tuple(baselines))
        from . import baselines as B
        for fw in fws:
            try:
                row[fw] = getattr(B, f"{fw}_baseline")(w, c, gb, S, R, precision)
            except ImportError:
                pass
            except Exception as e:  # noqa
                row[fw + "_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        if "apexnet" in row:
            for k in ("keras_default", "keras_tuned", "torch", "jax"):
                if k in row:
                    row[f"apexnet_vs_{k}"] = row["apexnet"] / row[k]
        rows.append(row)
        if verbose:
            f = lambda k: f"{row[k]:>10,.0f}" if k in row else "     error"
            r = row.get("apexnet_vs_keras_tuned")
            extra = "".join(f"  {fw} {f(fw)}" for fw in ("torch", "jax") if fw in row or fw + "_error" in row)
            print(f"  {w:>4} {precision:>8}  keras(default) {f('keras_default')}  keras(tuned) {f('keras_tuned')}{extra}  "
                  f"apexnet {f('apexnet')}  samples/s   apexnet/keras(tuned) = {'%.2fx' % r if r else 'n/a'}"
                  + "".join(f"  apexnet/{fw} = {row['apexnet_vs_' + fw]:.2f}x" for fw in ("torch", "jax") if 'apexnet_vs_' + fw in row), flush=True)
    return rows


# ======================================================================= run
def run(hw=None, quick=False, do_benchmark=True, report_path="apexnet_hw_report.json", prefer="auto", verbose=True):
    if hw is None:
        hw = H.setup(prefer=prefer, tf32=False, verbose=verbose)
    checks = [("fp32 equivalence vs CPU reference", check_equivalence), ("exact metrics under padded static batches", check_padding_exactness),
              ("TPU trace-mode model build", check_trace_build), ("mixed precision training", check_mixed_precision), ("XLA jit", check_xla), ("XLA fallback for uncompilable models", check_xla_fallback), ("multi-step loop (steps_per_call)", check_multi_step),
              ("LSTM / Transformer / BatchNorm-CNN smoke", check_models), ("checkpoint / export / save_model", check_checkpoint_export), ("PyTorch / JAX interop", check_interop)]
    results = []
    for name, fn in checks:
        t0 = time.time()
        try:
            detail, status = fn(hw), "PASS"
        except Skip as e:
            detail, status = str(e), "SKIP"
        except Exception as e:  # noqa
            detail, status = "".join(traceback.format_exception_only(type(e), e)).strip()[:400], "FAIL"
            if verbose:
                traceback.print_exc()
        results.append({"name": name, "status": status, "seconds": round(time.time() - t0, 1), "detail": detail})
        if verbose:
            print(f"[{status}] {name} ({results[-1]['seconds']}s): {detail}", flush=True)
    bench = []
    if do_benchmark and not quick:
        if verbose:
            print(f"\nBenchmark: ApexNet vs Keras on {hw.kind.upper()} x{hw.replicas}, precision={hw.precision}", flush=True)
        try:
            import keras  # noqa
            bench = benchmark(hw, verbose=verbose)
        except ImportError:
            print("keras not importable: benchmark skipped")
    try:
        import apexnet
        fv = apexnet.__version__
    except Exception:  # noqa
        fv = "?"
    try:
        import keras
        kv = keras.__version__
    except Exception:  # noqa
        kv = None
    report = {"apexnet": fv, "keras": kv, "hardware": hw.to_dict(), "checks": results, "benchmark": bench,
              "summary": {s: sum(r["status"] == s for r in results) for s in ("PASS", "FAIL", "SKIP")}}
    if report_path:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=1, default=float)
    if verbose:
        print("\nsummary:", report["summary"], "| report:", report_path)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-benchmark", action="store_true")
    ap.add_argument("--prefer", default="auto", choices=["auto", "cpu", "gpu", "tpu"])
    ap.add_argument("--report", default="apexnet_hw_report.json")
    a = ap.parse_args(argv)
    rep = run(quick=a.quick, do_benchmark=not a.no_benchmark, prefer=a.prefer, report_path=a.report)
    sys.exit(1 if rep["summary"]["FAIL"] else 0)


if __name__ == "__main__":
    main()
