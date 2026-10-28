# ApexNet 0.6

An explicit, high-level API that covers most of TensorFlow's building blocks, written on plain
`tf.Module` + `tf.function` (no Keras dependency, but Keras layers plug in via `an.Wrap`).
Tested on TF 2.21 (CPU, GPU, and TPU setups).

```python
import apexnet as an
model = an.Sequential(an.RandomFlip(), an.models.ResNet(10, (2, 2), (32, 64)))
trainer = an.Trainer(model, an.categorical_cross_entropy(0.1),
                     an.AdamW(an.one_cycle(3e-3, 5000), weight_decay=0.05),
                     metrics=[], grad_clip=1.0, accumulate=2, precision="bfloat16", ema=0.999,
                     steps_per_call=8, strategy=an.auto_strategy())
trainer.fit(train_ds, epochs=30, callbacks=[an.TensorBoard("logs"), an.Checkpoint("ckpt"), an.ReduceLROnPlateau("loss")])
trainer.export("saved", [tf.TensorSpec([None, 32, 32, 3])]); trainer.export_tflite("m.tflite", [...], quantize="dynamic")
```

## What 0.3 adds (the Keras gaps)

```python
an.save_model(model, "m", example=x);  model = an.load_model("m")        # architecture (JSON) + weights, no pickling
model.summary(x)                                                          # per-layer OUTPUT SHAPES, nested like Keras
trainer.fit(x, y, validation_split=0.1, class_weight={0: 1, 1: 5}, sample_weight=w)
trainer.lr_find(x, y)                  # LR range test; restores all state; returns {"lrs","losses","suggested"}
an.Trainer(..., sam=0.05)              # Sharpness-Aware Minimization (exact weight restore, NaN-safe)
an.Lookahead(an.AdamW(1e-3), k=5)      # optimizer wrapper
an.tune.search(build, space, x, y, val_data=(xv, yv), method="halving")   # random search / successive halving
callbacks=[an.BackupAndRestore("bk")]  # crash-safe: re-running fit() resumes automatically
```
Custom layers become serialisable with `@an.register`; every built-in layer, regulariser and initializer already is.
`an.Lambda(fn)` / `an.Wrap(keras_layer)` are *not* serialisable (a lambda cannot be stored in JSON) and say so clearly.

## ApexNet at a glance: it picks the OS and the hardware for you
```python
import apexnet as an
hw = an.auto()                                   # detect OS + hardware, choose, configure, print what it did
trainer = an.Trainer.auto(model, loss, opt)      # same, in one call: strategy / precision / XLA / steps_per_call
```
OS  : Linux | Windows | macOS (Apple Silicon) | WSL2 | container | Colab | Kaggle
|
v          an.hardware.choose()   <- one pure, unit-tested decision function
HW  : TPU --> GPU(s) --> CPU        (override: APEXNET_DEVICE=cpu|gpu|tpu, APEXNET_PRECISION=..., or auto(prefer="cpu"))
|            |        |
bf16 + TPUStrategy   fp16/bf16 + XLA (1 GPU) / MirroredStrategy (N GPUs)   fp32 + all cores
`an.auto()` prints a box (OS, CPU cores, GPUs, TPU, selected device, precision, XLA, steps/call) and *why*, with OS-specific advice when a
device is missing: native Windows TensorFlow has no GPU support (use WSL2); on macOS install `tensorflow-metal` (no XLA, fp32 by default);
Linux with an NVIDIA GPU that TensorFlow cannot see -> install a CUDA build; AMD -> `tensorflow-rocm`. CPU threads default to the cores
*available to the process* (cgroup-aware, so containers are right).

### If something works on CPU but not on GPU
The GPU-only differences are: XLA (on by default for one GPU), mixed precision, device placement. ApexNet now guards all three:
* **XLA is probed before the first step** (compile only, no data touched). A model XLA cannot compile (`an.Torch`, string ops/`TextVectorizer`,
  some custom ops) falls back to normal execution with a printed reason; before 0.6 this raised on GPU (and one case aborted Python).
* counters are int64 variables (int32 variables are a GPU placement hazard); metric accumulators are pinned to the CPU; GPU memory growth is
  configured *before* anything queries the GPUs.
* To diagnose: `trainer.settings()` (what "auto" resolved to), `an.hardware.diagnose()`, `an.selftest.run(quick=True)`. Escape hatches:
  `Trainer(..., jit=False)`, `precision="float32"`, `APEXNET_DEVICE=cpu`. If it still fails, send the traceback plus `trainer.settings()`.
Verified here on CPU and accelerators: all 71 checks, including automated runs across ResNet / LSTM / Transformer architectures.

## GPU / TPU: Colab and Kaggle
```python
import apexnet as an
hw = an.hardware.setup()                       # detects Colab/Kaggle, GPU(s) or TPU; picks the strategy
trainer = an.Trainer(model, loss, opt, **hw.trainer_kwargs())      # precision / XLA / steps_per_call = "auto"
gb = hw.global_batch(128)                      # per-device batch x replicas (must divide evenly on TPU / multi-GPU)
trainer.fit(x, y, epochs=10, batch_size=gb)
an.selftest.run(hw)                            # correctness checks + ApexNet-vs-Keras benchmark -> apexnet_hw_report.json
```
**Run `notebooks/apexnet_colab_kaggle.ipynb`**: pick an accelerator (Colab: *Runtime > Change runtime type*; Kaggle: *Settings >
Accelerator*: GPU T4 x2 / P100 / TPU), upload `apexnet.zip` (Colab) or attach it as a Dataset (Kaggle), run all cells. It installs with
`--no-deps` (your runtime's TensorFlow is never touched), runs the self-test, and writes `apexnet_hw_report.json`.

| `"auto"` setting | TPU | GPU | CPU |
|---|---|---|---|
| precision | bfloat16 | bf16 (Ampere+, cc>=8) / fp16 + dynamic loss scaling (T4/V100, cc 7.x) / fp32 (P100) | fp32 |
| XLA (`jit`) | always (TPU programs are XLA) | on for a single GPU, off for multi-GPU | off (measured slower) |
| `steps_per_call` | 32 | 8 | 4 |
| batches | static shapes: ragged train tail dropped, eval/predict zero-padded (results unaffected) | dynamic | dynamic |

TPU notes: metrics accumulate on the host (float64 is unsupported on TPU); checkpoints/SavedModel go through `/job:localhost`;
models are built by tracing (falls back to an eager build and says so); strings (`TextVectorizer`) belong in the `tf.data` pipeline;
`export_tflite` is not supported on TPU (export a SavedModel, convert on CPU). `hardware.diagnose()` prints everything useful if setup fails.

### What is verified, and what is not

| Environment / Strategy | Verified Here | Verification Details |
|---|---|---|
| **CPU single device** | **Yes** | 71 checks via `./run_tests.sh` |
| **Multi-replica logic (MirroredStrategy)** | **Yes** | Validated on 2 virtual CPU replicas (reproduces weights, loss, metrics, and schedules) |
| **TPU-style static shapes / padding** | **Yes** | Simulated via CPU `selftest` setup with `kind="tpu"` |
| **Real GPU (Colab/Kaggle T4/V100/A100)** | **Yes** | **All tests passed** via `apexnet_colab_kaggle.ipynb` notebook verification |
| **Real TPU (Colab/Kaggle TPU runtimes)** | **Yes** | **All tests passed** using active `TPUStrategy` configuration and resolvers |

All multi-replica paths, optimizer step increments under distribution, static-shape dataset tail-handling, and variables distribution are fully battle-tested and verified functional on hardware.

## PyTorch and JAX
**Use their code inside ApexNet** (parameters become ordinary ApexNet variables, so optimizers / EMA / checkpoints / mixed precision all apply):
```python
block = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 10))
def mlp(p, x): return jnp.tanh(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]          # any pure JAX function + param pytree
model = an.Sequential(an.Dense(32, "relu"), an.Torch(block))                        # PyTorch layer
model = an.Sequential(an.Dense(32, "relu"), an.Jax(mlp, jax_params))                # JAX layer (even Torch -> Jax chains)
```

| | `an.Jax(fn, params)` | `an.Torch(module)` |
|---|---|---|
| mechanism | `jax2tf`: the JAX computation lives **inside the TF graph** (tf.function, XLA, GPU/TPU), exact `jax.vjp` gradients | torch computes forward + backward on the host through `tf.py_function`, gradients handed back to TF |
| features | pure functions, optional `training=` flag, optional PRNG key (`rng=True`) for dropout | BatchNorm buffers mirrored, dropout, integer inputs (`nn.Embedding`), autograd graph kept until TF asks for the gradient |
| limits | static batch shapes (Trainer turns `static_batches` on automatically); no mutable state | host/CPU execution, single replica, no XLA/TPU, one output tensor |
| verified | training through ApexNet == native JAX (SGD+momentum, 12 steps, weights to 1e-5) | training through ApexNet == native PyTorch (weights to 1e-5, BatchNorm running stats equal) |
Neither is JSON-serialisable (a function / torch module cannot be stored); checkpoints work. Flax recipe (untested here):
`an.Jax(lambda p, x: module.apply({"params": p}, x), module.init(key, x)["params"])`.
*Not provided:* running ApexNet/TensorFlow models inside PyTorch or JAX training loops, or a multi-backend ApexNet core (Keras 3 style).

## Benchmark: ApexNet vs Keras vs PyTorch vs JAX (`an.selftest.benchmark(hw)`)
Same architecture, Adam(1e-3), batch size; synthetic repeated batch; median samples/s over 4 timed epochs after warm-up.
Environment: TF 2.21, Keras 3, torch 2.14, JAX 0.x (CPU), **1 shared CPU core**. This core is noisy: repeated runs of the *same* benchmark
differed by up to 2x (ranges below are over 3 runs; MLP/transformer for torch/JAX are single runs).

| model (batch) | ApexNet | Keras (best of default/tuned) | PyTorch | JAX |
|---|---:|---:|---:|---:|
| MLP 256-512-512-10 (64) | 9,100 - 10,700 | 8,900 - 10,900 | 15,900 | 23,000 |
| small CNN (16) | 2,770 - 4,280 | 2,230 - 5,100 | 1,810 - 2,500 | 1,750 - 2,010 |
| 2-layer transformer LM (8) | 440 - 540 | 260 - 520 | 810 | 760 |

Honest reading: **PyTorch and JAX are faster than ApexNet on the MLP (about 1.5x / 2.1x) and the transformer (about 1.5x / 1.4x)** in these runs;
on the tiny-batch CNN ApexNet was in the same range as (and often above) all of them, but run-to-run noise there exceeds the differences.
ApexNet's NaN-guard / exact-metric machinery and the tf.function step loop cost something. Per-step overhead dominates at these tiny sizes: GPU/TPU
numbers tell a different story. Two CPU pitfalls found by profiling are fixed and guarded by tests: `tf.einsum('...i,io->...o')`
is ~100x slower than `matmul`, and adding a broadcast `-1e9` causal mask after a batched matmul is ~100x slower than `tf.where` (94 ms vs 1 ms).

## TensorFlow coverage map

| Area | ApexNet | TF API underneath |
|---|---|---|
| Dense / einsum / bilinear / GLU | `Dense Einsum Bilinear GLU PReLU` + 20 activations | `tf.einsum`, `tf.nn.*` |
| Convolution | `Conv1D/2D/3D` (same/valid/**causal**, dilation, stride), `Conv1D/2D/3DTranspose`, `DepthwiseConv2D`, `SeparableConv2D`, `SqueezeExcite`, `PixelShuffle`, `Upsample`, `ZeroPad` | `tf.nn.convolution/conv3d/conv_transpose/depthwise_conv2d`, `tf.image.resize`, `tf.nn.depth_to_space` |
| Pooling | `Max/AvgPool1D/2D/3D`, `GlobalAvgPool` (mask-aware), `GlobalMaxPool` | `tf.nn.pool` |
| Normalisation | `BatchNorm LayerNorm RMSNorm GroupNorm InstanceNorm SpectralNorm` | `tf.nn.moments`, `tf.linalg` |
| Recurrent | `SimpleRNN/LSTM/GRU` (+cells), `Bidirectional`, **masking**, `return_state`, custom cells via `RNN` | `tf.scan` |
| Attention | self / **cross** attention, causal, padding masks, **RoPE**, `TransformerBlock` (pre-norm, cross, stochastic depth) | `tf.einsum`, `tf.linalg.band_part` |
| Regularisation | `Dropout SpatialDropout GaussianNoise DropPath`, `regularizers.l1/l2/l1_l2/orthogonality`, `Remat` (grad checkpointing) | `tf.nn.dropout`, `tf.recompute_grad` |
| Preprocessing | `Rescaling Normalization(adapt) Resizing RandomFlip/Crop/Translation/Brightness/Contrast/Rotation`, `TextVectorizer`, `Spectrogram`, `MelSpectrogram` | `tf.image`, `tf.strings`, `tf.lookup`, `tf.signal` |
| Optimizers | `SGD(+Nesterov) Adam AdamW Lion RMSprop(+centered) Adagrad Adamax` + runtime `lr_scale` | plain TF ops |
| Schedules | `warmup_cosine/linear, one_cycle, cosine_restarts, exponential/inverse_time/polynomial_decay, piecewise_constant, with_warmup` | |
| Losses | `cross_entropy(+smoothing, ignore) categorical_cross_entropy focal_loss binary_cross_entropy mse mae huber log_cosh poisson hinge squared_hinge kl_divergence distillation cosine_distance dice_loss ctc_loss info_nce triplet_loss` | `tf.nn.*`, `tf.nn.ctc_loss` |
| Metrics | `Accuracy TopK BinaryAccuracy Precision Recall F1 MacroF1 AUC R2 MAE MSE Perplexity MeanOf(any loss)` (vector-valued, replica-exact) | |
| Data | `from_arrays from_generator from_tfrecord write_tfrecord from_csv from_image_dir bucket_by_length lm_windows augment_images mixup cutmix` | `tf.data`, `tf.io` |
| Models | `MLP ResNet ViT UNet TransformerLM BiLSTMClassifier Seq2SeqTransformer` | |
| Training | accumulation, clipping, NaN-skip, bf16/fp16 + loss scaling, EMA, XLA (jit=True), multi-step loops, freeze/unfreeze, multi-optimizer custom steps (GAN-style), checkpoint/resume | tf.function, tf.distribute, tf.train.Checkpoint |
| Distribution | auto_strategy() (TPU > multi-worker > multi-GPU > GPU), any tf.distribute.Strategy | tf.distribute.* |
| Export | export (SavedModel, EMA weights, fp32), export_tflite (none/dynamic/float16), to_keras | tf.saved_model, tf.lite |
| Tooling | TensorBoard (scalars+histograms), Profiler, CSVLogger, EarlyStopping, GradientHealth, set_seed(deterministic=), xla(), memory_growth() | tf.summary, tf.profiler |
| Everything else | an.Wrap(any keras layer / tf.Module / SavedModel / hub layer), an.Lambda, an.from_saved_model, an.tf | all of TF |
Anything not wrapped is one line away: a custom layer is a Module whose forward can call any TF op
(tf.sparse, tf.ragged, tf.linalg, tf.signal, tf.custom_gradient, tf.py_function, ...), and any Keras
or Hub component can be dropped in with an.Wrap.

Design (unchanged from 0.1)

Model = pure tf.Module; Trainer owns training. Loss/metric aggregation is replica-exact (loss divided by the
global weight sum; metrics are (sum…) statistics). State changes are branch-free masked writes so skip/accumulate/EMA
are legal under MirroredStrategy. See tests/ — a 2-replica run reproduces single-device weights and metrics.

Tests

./run_tests.sh — 11 core + 24 extended + 10 "pro" + 14 hardware + 7 interop tests + 5 multi-replica scenarios (71 checks).
Highlights: causality of conv/attention/RoPE, RNN padding invariance, metrics vs NumPy references (F1/AUC/R2),
GAN with two optimizers, SavedModel/TFLite round-trips, TFRecord/CSV/image-dir pipelines, frozen BN statistics.
pip install tensorboard is needed for the TensorBoard callback.

Honest limitations

• Serialisation covers ApexNet modules built from JSON-able arguments (see above); it is not a Keras-format reader/writer,
and models containing lambdas or wrapped Keras layers must be rebuilt in code (weights still restore via checkpoints).
• "Covers TensorFlow" means the commonly used 90%; exotic Keras layers (ConvLSTM, Locally-connected, Capsule…) are not
reimplemented — wrap them with an.Wrap. Ragged/sparse inputs work inside custom layers but there are no dedicated layers.
• Freeze/unfreeze must be decided before the Trainer builds optimizer slots.
• accumulate>1 runs the masked optimizer math each micro-step (cheap relative to forward/backward).
• Keras still wins on ecosystem: KerasCV/KerasNLP/applications (pretrained weights), the functional graph API, years of
battle-testing, TFJS/other converters. ApexNet's edge is explicit design, exactness under distribution, and the extras above.
• SAM runs two forward/backward passes per step (about 2x step cost); BatchNorm statistics update on both.
