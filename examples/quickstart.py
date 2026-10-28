here"""Train a tiny character-level Transformer LM, with the full ApexNet feature set switched on."""
import numpy as np
import apexnet as an

text = ("the quick brown fox jumps over the lazy dog. " * 300)
chars = sorted(set(text)); idx = {c: i for i, c in enumerate(chars)}
tokens = np.array([idx[c] for c in text], dtype="int32")

SEQ = 32
train = an.data.lm_windows(tokens, SEQ, batch_size=32, seed=0)

model = an.models.TransformerLM(len(chars), dim=64, depth=2, heads=4, max_len=SEQ)
steps = 6 * 40
trainer = an.Trainer(
    model,
    loss=an.cross_entropy(label_smoothing=0.0),
    optimizer=an.AdamW(an.warmup_cosine(3e-3, 20, steps), weight_decay=0.01),
    metrics=[an.Perplexity(), an.Accuracy()],
    grad_clip=1.0,          # global-norm clipping
    accumulate=2,           # effective batch = 64
    precision="bfloat16",   # bf16 math, fp32 weights
    ema=0.99,               # evaluate/predict with EMA weights automatically
    steps_per_call=4,       # 4 train steps per tf.function call
)
trainer.fit(train, epochs=6, callbacks=[an.EarlyStopping("loss", patience=3)])
trainer.summary()
print("eval:", trainer.evaluate(train))
