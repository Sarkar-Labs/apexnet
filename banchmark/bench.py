"""ApexNet vs Keras head-to-head: same architecture, optimizer, data, batch size. Reports steady-state
training throughput (samples/s) after a warm-up epoch, best of N timed epochs. Usage: bench.py {mlp,cnn,tfm}"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import keras
import apexnet as an

which = sys.argv[1] if len(sys.argv) > 1 else "mlp"
JIT = len(sys.argv) > 2 and sys.argv[2] == "jit"   # compare XLA-compiled Keras vs XLA-compiled ApexNet
rs = np.random.RandomState(0)
BS, EPOCHS = 128, 3


def make(which):
    if which == "mlp":
        N = 25600
        x = rs.randn(N, 256).astype("float32"); y = rs.randint(0, 10, N).astype("int32")
        k = keras.Sequential([keras.layers.Input((256,)), keras.layers.Dense(512, activation="relu"),
                              keras.layers.Dense(512, activation="relu"), keras.layers.Dense(10)])
        f = an.Sequential(an.Dense(512, "relu"), an.Dense(512, "relu"), an.Dense(10))
        loss_k = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    elif which == "cnn":
        N = 3840
        x = rs.randn(N, 32, 32, 3).astype("float32"); y = rs.randint(0, 10, N).astype("int32")
        L = keras.layers
        k = keras.Sequential([L.Input((32, 32, 3)), L.Conv2D(32, 3, padding="same", activation="relu"), L.MaxPool2D(2),
                              L.Conv2D(64, 3, padding="same", activation="relu"), L.MaxPool2D(2),
                              L.GlobalAveragePooling2D(), L.Dense(10)])
        f = an.Sequential(an.Conv2D(32, 3, activation="relu"), an.MaxPool2D(2), an.Conv2D(64, 3, activation="relu"),
                          an.MaxPool2D(2), an.GlobalAvgPool(), an.Dense(10))
        loss_k = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:  # tiny transformer LM
        N, T, V = 1280, 64, 256
        x = rs.randint(0, V, (N, T)).astype("int32"); y = rs.randint(0, V, (N, T)).astype("int32")
        L = keras.layers
        inp = keras.Input((T,), dtype="int32"); h = L.Embedding(V, 64)(inp)
        for _ in range(2):
            a = L.MultiHeadAttention(4, 16)(L.LayerNormalization()(h), L.LayerNormalization()(h), use_causal_mask=True)
            h = h + a
            h = h + L.Dense(64)(L.Dense(256, activation="gelu")(L.LayerNormalization()(h)))
        k = keras.Model(inp, L.Dense(V)(L.LayerNormalization()(h)))
        f = an.models.TransformerLM(V, 64, 2, 4, T)
        loss_k = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return x, y, k, f, loss_k, N


def timed(fit_epoch, n):
    print("  timing...", flush=True)
    fit_epoch()  # warm-up (tracing / XLA / first-batch init)
    best = min((lambda t0: (fit_epoch(), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(EPOCHS))
    return n / best


x, y, k, f, loss_k, N = make(which)
res = {}
for spe in (1, 16):
    kk = keras.models.clone_model(k)
    kk.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss_k, steps_per_execution=spe, jit_compile=JIT)
    res[f"keras spe={spe}"] = timed(lambda: kk.fit(x, y, batch_size=BS, epochs=1, verbose=0, shuffle=False), N)
ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(BS).prefetch(tf.data.AUTOTUNE)
for spc in (1, 16):
    tr = an.Trainer(f, an.cross_entropy(), an.Adam(1e-3), steps_per_call=spc, jit=JIT)
    res[f"apexnet spc={spc}"] = timed(lambda: tr.fit(ds, epochs=tr.epoch + 1, verbose=0), N)
print(which, "XLA" if JIT else "eager-graph", {k_: round(v) for k_, v in res.items()}, "samples/s")￼Enter
