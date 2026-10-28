here"""Many ApexNet/TensorFlow features in one place (synthetic data so it runs anywhere in ~1 min):
augmentation layers, ResNet, mixup + soft labels, EMA, bf16, one-cycle LR, F1/AUC-style metrics,
TensorBoard, plateau LR, checkpointing, SavedModel + TFLite export."""
import tempfile, os
import numpy as np
import tensorflow as tf
import apexnet as an

an.set_seed(0)
rs = np.random.RandomState(0)
N, C = 512, 3
y = rs.randint(0, C, N).astype("int32")
x = rs.randn(N, 16, 16, 3).astype("float32") * 0.3
for c in range(C):                      # class c = bright stripe in row-band c
    x[y == c, c * 5:c * 5 + 4] += 1.0

model = an.Sequential(
    an.RandomFlip("horizontal"), an.RandomTranslation(2), an.RandomBrightness(0.1),   # train-only augmentation
    an.models.ResNet(C, blocks=(1, 1), widths=(16, 32)),
)
steps = 8 * (N // 64)
trainer = an.Trainer(
    model, loss=an.categorical_cross_entropy(label_smoothing=0.05),
    optimizer=an.AdamW(an.one_cycle(3e-3, steps), weight_decay=0.01),
    metrics=[], grad_clip=1.0, precision="bfloat16", ema=0.99, steps_per_call=2)
train = an.data.from_arrays(x, y, batch_size=64, shuffle=True, drop_remainder=True, seed=0).map(an.data.mixup(C, 0.2))
val = an.data.from_arrays(x[:128], y[:128], batch_size=64)

# validation uses hard labels -> a separate evaluator sharing the same model & EMA weights
evaluator = an.Trainer(model, an.cross_entropy(), an.SGD(0.0), metrics=[an.Accuracy(), an.MacroF1(C)])
d = tempfile.mkdtemp()
trainer.fit(train, epochs=8, callbacks=[an.TensorBoard(d + "/tb"), an.Checkpoint(d + "/ckpt", keep=2),
                                        an.ReduceLROnPlateau("loss", patience=3), an.GradientHealth()])
evaluator.init(x[:2])
with trainer.ema_weights():
    print("val (EMA weights):", evaluator.evaluate(val, use_ema=False))
trainer.export(d + "/saved", [tf.TensorSpec([None, 16, 16, 3], tf.float32)])
trainer.export_tflite(d + "/model.tflite", [tf.TensorSpec([1, 16, 16, 3], tf.float32)], quantize="dynamic")
print("exported:", sorted(os.listdir(d)), "tflite bytes:", os.path.getsize(d + "/model.tflite"))
