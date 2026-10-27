"""Environment helpers: seeding, determinism, devices, XLA, memory growth, mixed-precision, strategies."""
import os
import random
import sys
import numpy as np
import tensorflow as tf


def set_seed(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if "torch" in sys.modules:           # also seed PyTorch when it is in use (an.Torch layers)
        sys.modules["torch"].manual_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def memory_growth(enable=True):
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, enable)


def xla(enable=True):
    """Global XLA auto-clustering."""
    tf.config.optimizer.set_jit(enable)


def devices():
    return {k: len(tf.config.list_physical_devices(k)) for k in ("CPU", "GPU", "TPU")}


def virtual_cpus(n=2):
    """Split the host CPU into n logical devices (to test multi-replica code anywhere). Call before TF init."""
    cpu = tf.config.list_physical_devices("CPU")[0]
    tf.config.set_logical_device_configuration(cpu, [tf.config.LogicalDeviceConfiguration()] * n)


def auto_strategy(tpu=None):
    """Best tf.distribute strategy for the environment (see hardware.setup for the full configuration)."""
    from .hardware import setup
    return setup(tpu=tpu, verbose=False).strategy
