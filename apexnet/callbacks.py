"""Callbacks. Hooks receive the Trainer, so they can read/modify anything."""
import math
import os
import time


class Callback:
    def on_fit_start(self, t): ...
    def on_epoch_start(self, t, epoch): ...
    def on_step_end(self, t, step): ...
    def on_epoch_end(self, t, epoch, logs): ...
    def on_fit_end(self, t): ...


class CallbackList:
    def __init__(self, cbs):
        self.cbs = list(cbs)

    def __getattr__(self, name):
        def call(*a):
            for c in self.cbs:
                getattr(c, name)(*a)
        return call


def _fmt(k, v):
    return f"{k} {v:.4g}" if abs(v) >= 1e-3 or v == 0 else f"{k} {v:.3e}"


class Logger(Callback):
    def __init__(self, every=0):
        self.every, self.t0 = every, 0.0

    def on_epoch_start(self, t, epoch):
        self.t0 = time.time()

    def on_step_end(self, t, step):
        if self.every and step % self.every == 0:
            r = t.train_meter.result(t.metrics, t.loss_name)
            print(f"  step {step}: " + " | ".join(_fmt(k, v) for k, v in r.items()), flush=True)

    def on_epoch_end(self, t, epoch, logs):
        print(f"epoch {epoch + 1}/{t.target_epochs} | " + " | ".join(_fmt(k, v) for k, v in logs.items())
              + f" | {time.time() - self.t0:.1f}s", flush=True)


class EarlyStopping(Callback):
    def __init__(self, monitor="val_loss", patience=3, mode="min", min_delta=0.0, restore_best=True):
        self.monitor, self.patience, self.mode = monitor, patience, mode
        self.min_delta, self.restore, self.best, self.wait, self.weights = min_delta, restore_best, None, 0, None

    def _better(self, v):
        if self.best is None:
            return True
        return v < self.best - self.min_delta if self.mode == "min" else v > self.best + self.min_delta

    def on_epoch_end(self, t, epoch, logs):
        v = logs[self.monitor]
        if self._better(v):
            self.best, self.wait = v, 0
            if self.restore:
                self.weights = t.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                t.stop = True

    def on_fit_end(self, t):
        if self.restore and self.weights is not None:
            t.set_weights(self.weights)


class Checkpoint(Callback):
    """Saves full training state (weights, optimizer, EMA, epoch) with rotation."""

    def __init__(self, directory, monitor=None, mode="min", keep=3):
        self.dir, self.monitor, self.mode, self.keep = directory, monitor, mode, keep
        self.best, self.paths = None, []

    def on_epoch_end(self, t, epoch, logs):
        if self.monitor:
            v = logs[self.monitor]
            if self.best is not None and ((v >= self.best) if self.mode == "min" else (v <= self.best)):
                return
            self.best = v
            path = os.path.join(self.dir, "best")
            t.save(path)
            return
        path = os.path.join(self.dir, f"epoch_{epoch + 1}")
        t.save(path)
        self.paths.append(path)
        while len(self.paths) > self.keep:
            old = self.paths.pop(0)
            for f in os.listdir(self.dir):
                if f.startswith(os.path.basename(old) + "."):
                    os.remove(os.path.join(self.dir, f))


class TerminateOnNaN(Callback):
    def on_epoch_end(self, t, epoch, logs):
        if any(math.isnan(v) or math.isinf(v) for v in logs.values()):
            print("non-finite value in logs, stopping")
            t.stop = True


class LambdaCallback(Callback):
    def __init__(self, on_epoch_end=None, on_step_end=None):
        self._e, self._s = on_epoch_end, on_step_end

    def on_epoch_end(self, t, epoch, logs):
        if self._e:
            self._e(t, epoch, logs)

    def on_step_end(self, t, step):
        if self._s:
            self._s(t, step)


class TensorBoard(Callback):
    """Scalars every epoch (and every ``every`` steps), optional weight histograms, and the graph-free
    hparams text. View with `tensorboard --logdir <dir>`."""

    def __init__(self, logdir="logs", histograms=False, every=0):
        import tensorflow as tf
        self.tf, self.dir, self.hist, self.every = tf, logdir, histograms, every
        self.writer = tf.summary.create_file_writer(logdir)

    def on_step_end(self, t, step):
        if self.every and step % self.every == 0:
            r = t.train_meter.result(t.metrics, t.loss_name)
            with self.writer.as_default(step=int(t.main_opt.iterations.numpy())):
                for k, v in r.items():
                    if isinstance(v, float):
                        self.tf.summary.scalar("step/" + k, v)

    def on_epoch_end(self, t, epoch, logs):
        with self.writer.as_default(step=epoch + 1):
            for k, v in logs.items():
                self.tf.summary.scalar(f"epoch/{k}", v)
            if self.hist:
                for v in t.model.trainable_variables:
                    self.tf.summary.histogram("weights/" + v.name.split(":")[0], v)
        self.writer.flush()

    def on_fit_end(self, t):
        self.writer.flush()


class ReduceLROnPlateau(Callback):
    """Multiply the optimizer's runtime lr_scale by ``factor`` when ``monitor`` stops improving."""

    def __init__(self, monitor="val_loss", factor=0.5, patience=2, mode="min", min_scale=1e-3, min_delta=1e-4):
        self.monitor, self.factor, self.patience, self.mode = monitor, factor, patience, mode
        self.min_scale, self.min_delta, self.best, self.wait = min_scale, min_delta, None, 0

    def on_epoch_end(self, t, epoch, logs):
        v = logs[self.monitor]
        better = self.best is None or (v < self.best - self.min_delta if self.mode == "min" else v > self.best + self.min_delta)
        if better:
            self.best, self.wait = v, 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            new = max(float(t.main_opt.lr_scale.numpy()) * self.factor, self.min_scale)
            for o in t.optimizers:
                o.lr_scale.assign(new)
            self.wait = 0
            print(f"  ReduceLROnPlateau: lr_scale -> {new:.4g}")


class CSVLogger(Callback):
    def __init__(self, path):
        self.path, self.keys = path, None

    def on_epoch_end(self, t, epoch, logs):
        import csv
        first = self.keys is None
        self.keys = self.keys or ["epoch"] + list(logs)
        with open(self.path, "w" if first else "a", newline="") as f:
            w = csv.writer(f)
            if first:
                w.writerow(self.keys)
            w.writerow([epoch + 1] + [logs.get(k, "") for k in self.keys[1:]])


class Profiler(Callback):
    """tf.profiler trace between two training calls (view in TensorBoard's Profile tab)."""

    def __init__(self, logdir="logs/profile", start_step=2, stop_step=6):
        self.dir, self.a, self.b, self.on = logdir, start_step, stop_step, False

    def on_step_end(self, t, step):
        import tensorflow as tf
        if step == self.a and not self.on:
            tf.profiler.experimental.start(self.dir)
            self.on = True
        elif step >= self.b and self.on:
            tf.profiler.experimental.stop()
            self.on = False

    def on_fit_end(self, t):
        import tensorflow as tf
        if self.on:
            tf.profiler.experimental.stop()
            self.on = False


class GradientHealth(Callback):
    """Warns when the epoch-end global grad-norm explodes or NaN steps were skipped."""

    def __init__(self, max_norm=1e3):
        self.max = max_norm

    def on_epoch_end(self, t, epoch, logs):
        if logs.get("grad_norm", 0) > self.max:
            print(f"  warning: grad_norm {logs['grad_norm']:.3g} > {self.max}")
        if logs.get("skipped"):
            print(f"  warning: {int(logs['skipped'])} non-finite steps skipped so far")


class LearningRateScheduler(Callback):
    """``fn(epoch, current_scale) -> new lr_scale`` applied at the start of every epoch (runtime multiplier on
    top of the optimizer's own schedule)."""

    def __init__(self, fn):
        self.fn = fn

    def on_epoch_start(self, t, epoch):
        new = float(self.fn(epoch, float(t.main_opt.lr_scale.numpy())))
        for o in t.optimizers:
            o.lr_scale.assign(new)


class BackupAndRestore(Callback):
    """Crash/pre-emption safety: saves the full training state every epoch and, when ``fit`` is called again,
    resumes from it automatically (same call, same epochs target)."""

    def __init__(self, directory):
        self.dir = directory
        self.path = os.path.join(directory, "latest")

    def on_fit_start(self, t):
        if os.path.exists(self.path + ".index"):
            t.load(self.path)
            print(f"resumed from {self.path} at epoch {t.epoch}")

    def on_epoch_end(self, t, epoch, logs):
        os.makedirs(self.dir, exist_ok=True)
        t.save(self.path)
