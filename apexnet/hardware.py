"""Hardware detection and one-call setup for local machines, Google Colab and Kaggle (CPU / GPU / multi-GPU / TPU).

    hw = an.hardware.setup()                       # detect, configure, pick a strategy
    trainer = an.Trainer(model, loss, opt, **hw.trainer_kwargs())   # precision / jit / steps_per_call / strategy
    ds = an.data.from_arrays(x, y, batch_size=hw.global_batch(128), drop_remainder=hw.kind == "tpu")

NOTE: the CPU paths are exercised by this project's test-suite. The GPU / TPU paths follow TensorFlow's documented
APIs but can only be validated on real hardware: run ``an.selftest.run()`` in Colab / Kaggle (see the notebook).
"""
import glob
import json
import os
import platform as _platform
import shutil
import sys
import tensorflow as tf


# ----------------------------------------------------------------- environment
def platform_name():
    """'colab' | 'kaggle' | 'local'"""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/working"):
        return "kaggle"
    if "google.colab" in sys.modules or any(k.startswith("COLAB_") for k in os.environ):
        return "colab"
    return "local"


def tpu_hint():
    """Cheap, non-invasive check for 'a TPU is probably attached' (never touches the TPU)."""
    return bool(os.environ.get("COLAB_TPU_ADDR") or os.environ.get("TPU_NAME") or os.environ.get("TPU_ADDRESS")
                or glob.glob("/dev/accel*"))


def gpu_info():
    out = []
    for g in tf.config.list_physical_devices("GPU"):
        try:
            d = tf.config.experimental.get_device_details(g)
        except Exception:  # noqa
            d = {}
        out.append({"name": d.get("device_name", g.name), "compute_capability": d.get("compute_capability")})
    return out


def recommend_precision(kind, compute_capability=None, os_name=None):
    """TPU -> bfloat16. GPU: Ampere+ (cc>=8.0) bfloat16, Volta/Turing (7.x, e.g. T4) float16 with loss scaling, older
    (e.g. P100) float32. macOS (Apple GPU via the Metal plugin) -> float32: mixed precision is not validated there.
    CPU -> float32."""
    if kind == "tpu":
        return "bfloat16"
    if kind == "gpu" and os_name != "macos" and compute_capability:
        if tuple(compute_capability) >= (8, 0):
            return "bfloat16"
        if tuple(compute_capability) >= (7, 0):
            return "float16"
    return "float32"


def recommend_steps_per_call(kind):
    """Host round-trips dominate on fast accelerators: run several steps per tf.function call (a few even on CPU)."""
    return {"tpu": 32, "gpu": 8}.get(kind, 4)


# ------------------------------------------------------------------------ OS
def os_info():
    """Operating-system facts that change how TensorFlow should be configured."""
    sysname = _platform.system()
    name = {"Darwin": "macos"}.get(sysname, sysname.lower())          # 'linux' | 'windows' | 'macos'
    machine = _platform.machine().lower()
    try:
        cores = len(os.sched_getaffinity(0))                         # respects cgroup / affinity limits (containers)
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    return {"system": name, "release": _platform.release(), "machine": machine,
            "arm": machine in ("arm64", "aarch64"),
            "apple_silicon": name == "macos" and machine in ("arm64", "aarch64"),
            "wsl": name == "linux" and ("microsoft" in _platform.release().lower() or "WSL_DISTRO_NAME" in os.environ),
            "container": os.path.exists("/.dockerenv"), "cpu_cores": cores, "python": sys.version.split()[0],
            "hosted": platform_name(), "nvidia_smi": shutil.which("nvidia-smi") is not None,
            "rocm_smi": shutil.which("rocm-smi") is not None}


class Plan:
    """The outcome of the (pure) hardware decision: which device kinds to try, in order, and why."""
    def __init__(self, order, reasons, notes, os_name, jit_ok):
        self.order, self.reasons, self.notes, self.os_name, self.jit_ok = order, reasons, notes, os_name, jit_ok

    @property
    def kind(self):
        return self.order[0]

    def __repr__(self):
        return f"Plan(order={self.order}, reasons={self.reasons}, notes={self.notes})"


def choose(devices, osi, prefer="auto", env=None):
    """THE hardware/OS decision, as a pure function (no TensorFlow calls), so every scenario is unit-testable.

    devices: {"gpu": [{"name", "compute_capability"}...], "tpu": bool (a TPU is hinted/requested)}
    osi:     os_info() dict.   prefer: 'auto' | 'cpu' | 'gpu' | 'tpu'   (env APEXNET_DEVICE overrides it)
    Priority for 'auto': TPU > GPU(s) > CPU; an explicit choice is strict (except that CPU is always the last resort).
    OS rules: macOS -> Apple GPU needs the Metal plugin, no XLA, fp32; native Windows -> CPU-only TensorFlow (use WSL2 for
    GPUs); Linux -> CUDA/ROCm builds; WSL/containers noted; Colab/Kaggle recognised."""
    env = os.environ if env is None else env
    prefer = (env.get("APEXNET_DEVICE") or prefer or "auto").lower()
    if prefer not in ("auto", "cpu", "gpu", "tpu"):
        raise ValueError(f"unknown device {prefer!r}; use auto|cpu|gpu|tpu")
    gpus, osn = devices.get("gpu") or [], osi["system"]
    reasons, notes = [], []
    if prefer == "auto":
        order = (["tpu"] if devices.get("tpu") else []) + (["gpu"] if gpus else []) + ["cpu"]
        reasons.append("auto: " + (f"TPU detected" if order[0] == "tpu" else f"{len(gpus)} GPU(s) detected" if order[0] == "gpu"
                                    else "no accelerator detected, using the CPU"))
    else:
        order = [prefer] if prefer == "cpu" else [prefer, "cpu"]
        reasons.append(f"{prefer.upper()} requested" + (" (env APEXNET_DEVICE)" if env.get("APEXNET_DEVICE") else ""))
    # ---- OS-specific guidance (notes explain WHY a device was not found / what to install)
    if not gpus and prefer in ("auto", "gpu"):
        if osn == "windows":
            notes.append("native Windows TensorFlow (>=2.11) has no GPU support: use WSL2 or a Linux/Colab runtime for GPUs"
                         + (" (an NVIDIA GPU is present)" if osi.get("nvidia_smi") else ""))
        elif osn == "macos" and osi.get("apple_silicon"):
            notes.append("Apple GPU not visible: pip install tensorflow-metal (needs a compatible TensorFlow)")
        elif osn == "linux" and osi.get("nvidia_smi"):
            notes.append("an NVIDIA GPU is present but this TensorFlow cannot see it: install a CUDA build "
                         "(pip install 'tensorflow[and-cuda]') or check the driver")
        elif osn == "linux" and osi.get("rocm_smi"):
            notes.append("an AMD GPU is present: install tensorflow-rocm")
    if osi.get("wsl"):
        notes.append("running under WSL2")
    if osi.get("container"):
        notes.append(f"container: {osi['cpu_cores']} CPU core(s) available to this process")
    if osn == "macos" and gpus:
        notes.append("macOS GPU (Metal): XLA is disabled and float32 is used by default")
    return Plan(order, reasons, notes, osn, jit_ok=(osn != "macos"))


# -------------------------------------------------------------------- TPU
def connect_tpu(tpu=None):
    """Try the resolver forms used by Colab / Kaggle TPU runtimes (TPU VM 'local', legacy gRPC address, auto-discovery).
    Returns (strategy, how). Raises RuntimeError with every attempt's error if none work."""
    cands = []
    if tpu:
        cands.append(tpu)
    for env in ("TPU_NAME", "TPU_ADDRESS"):
        if os.environ.get(env):
            cands.append(os.environ[env])
    if os.environ.get("COLAB_TPU_ADDR"):
        cands.append("grpc://" + os.environ["COLAB_TPU_ADDR"])
    cands += ["local", "connect", ""]
    seen, errors = set(), []
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        try:
            R = tf.distribute.cluster_resolver.TPUClusterResolver
            resolver = R.connect() if c == "connect" else (R(tpu=c) if c else R())
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            return tf.distribute.TPUStrategy(resolver), c or "auto"
        except Exception as e:  # noqa
            errors.append(f"  tpu={c!r}: {type(e).__name__}: {str(e)[:160]}")
    raise RuntimeError("could not initialise a TPU. Attempts:\n" + "\n".join(errors) + "\n"
                       "Check the runtime type (TPU selected?) and that this TensorFlow build supports TPUs; "
                       "run an.hardware.diagnose() for details.")


def checkpoint_options(strategy):
    """TPU workers may be remote: write/read checkpoints through the local host."""
    if "TPU" in type(strategy).__name__:
        return tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    return None


# ------------------------------------------------------------------- Hardware
class Hardware:
    def __init__(self, kind, strategy, platform, gpus, how=None, notes=(), osinfo=None, reasons=()):
        self.kind, self.strategy, self.platform, self.gpus, self.how = kind, strategy, platform, gpus, how
        self.notes, self.reasons = list(notes), list(reasons)
        self.os = osinfo or os_info()
        self.replicas = strategy.num_replicas_in_sync
        cc = gpus[0]["compute_capability"] if gpus else None
        self.precision = recommend_precision(kind, cc, self.os["system"])
        self.steps_per_call = recommend_steps_per_call(kind)

    def global_batch(self, per_replica):
        """Global batch for a desired per-device batch (TPU/multi-GPU need divisibility by the replica count)."""
        return per_replica * self.replicas

    def trainer_kwargs(self, precision="auto", steps_per_call="auto", jit="auto"):
        return {"strategy": self.strategy, "precision": precision, "steps_per_call": steps_per_call, "jit": jit}

    def scope(self):
        return self.strategy.scope()

    def to_dict(self):
        return {"platform": self.platform, "kind": self.kind, "replicas": self.replicas,
                "strategy": type(self.strategy).__name__, "tpu_connection": self.how, "gpus": self.gpus,
                "recommended_precision": self.precision, "recommended_steps_per_call": self.steps_per_call,
                "tensorflow": tf.__version__, "python": sys.version.split()[0], "os": _platform.platform(),
                "os_info": self.os, "why": self.reasons, "notes": self.notes}

    def describe(self):
        return describe(self)

    def __repr__(self):
        g = "; ".join(f"{x['name']} (cc {x['compute_capability']})" for x in self.gpus) or "-"
        return (f"Hardware(os={self.os['system']}/{self.os['machine']}, platform={self.platform}, kind={self.kind}, "
                f"replicas={self.replicas}, strategy={type(self.strategy).__name__}, gpus=[{g}], precision={self.precision}, "
                f"steps_per_call={self.steps_per_call})")


def _make_strategy(kind, tpu, gpus_phys, notes):
    """-> (strategy, how). Raises RuntimeError if this kind is unavailable."""
    if kind == "tpu":
        return connect_tpu(tpu)
    if kind == "gpu":
        if not gpus_phys:
            raise RuntimeError("no GPU found (Colab: Runtime > Change runtime type; Kaggle: Settings > Accelerator)")
        return (tf.distribute.MirroredStrategy() if len(gpus_phys) > 1 else tf.distribute.get_strategy()), None
    return tf.distribute.get_strategy(), None


def setup(prefer="auto", precision="auto", tpu=None, seed=None, memory_growth=True, tf32=None, xla=None,
          threads="auto", verbose=True):
    """Detect the OS and the hardware, pick the device (TPU > GPU(s) > CPU, see ``choose``) and configure TensorFlow.
    ``prefer``: 'auto' | 'cpu' | 'gpu' | 'tpu' (env ``APEXNET_DEVICE`` / ``APEXNET_PRECISION`` override).
    ``threads``: 'auto' = use every CPU core available to this process (cgroup-aware) for intra/inter-op parallelism.
    ``tf32``: None (TF default) / True / False (False = strict fp32 matmuls on Ampere+, needed for exactness tests).
    Must run before other TensorFlow GPU work (memory growth cannot change after the GPU is initialised)."""
    osi = os_info()
    gpus_phys = tf.config.list_physical_devices("GPU")
    notes = []
    if memory_growth:   # must come first: querying device details can initialise the GPUs, after which this is impossible
        for g in gpus_phys:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError as e:  # GPUs already initialised
                notes.append(f"memory growth not set: {e}")
    plan = choose({"gpu": gpu_info() if gpus_phys else [], "tpu": bool(tpu) or tpu_hint()}, osi, prefer)
    notes, reasons = notes + list(plan.notes), list(plan.reasons)
    if tf32 is not None:
        tf.config.experimental.enable_tensor_float_32_execution(bool(tf32))
    if xla is not None:
        tf.config.optimizer.set_jit(bool(xla))
    if threads == "auto" and plan.kind == "cpu" and osi["cpu_cores"] > 1:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(osi["cpu_cores"])
            tf.config.threading.set_inter_op_parallelism_threads(max(1, osi["cpu_cores"] // 2))
            reasons.append(f"CPU threads: {osi['cpu_cores']} intra-op")
        except RuntimeError:
            notes.append("CPU thread counts could not be set (TensorFlow already initialised)")
    if seed is not None:
        from .utils import set_seed
        set_seed(seed)
    strategy, kind, how = None, None, None
    for k in plan.order:
        try:
            strategy, how = _make_strategy(k, tpu, gpus_phys, notes)
            kind = k
            break
        except RuntimeError as e:
            if len(plan.order) == 1 or k == plan.order[-1]:
                raise
            notes.append(f"{k.upper()} unavailable ({str(e).splitlines()[0][:110]}); trying the next device")
    hw = Hardware(kind, strategy, platform_name(), gpu_info() if kind == "gpu" else [], how, notes, osi, reasons)
    prec = os.environ.get("APEXNET_PRECISION") or precision
    if prec != "auto":
        hw.precision = prec
    if verbose:
        print(describe(hw))
    return hw


def auto(**kw):
    """Shorthand: ``hw = an.auto()`` == ``an.hardware.setup()``."""
    return setup(**kw)


def describe(hw):
    """A boxed, human-readable summary of what ApexNet detected and chose."""
    o = hw.os
    gpu = "; ".join(f"{g['name']}" + (f" (cc {g['compute_capability'][0]}.{g['compute_capability'][1]})" if g.get("compute_capability") else "")
                    for g in hw.gpus) or "-"
    rows = [("OS", f"{o['system']} {o['machine']}" + (" (WSL2)" if o["wsl"] else "") + (" (container)" if o["container"] else "")
             + (f"  [{o['hosted']}]" if o["hosted"] != "local" else "")),
            ("CPU cores", str(o["cpu_cores"])), ("GPU", gpu), ("TPU", "yes (" + str(hw.how) + ")" if hw.kind == "tpu" else "-"),
            ("selected", f"{hw.kind.upper()} x{hw.replicas}  ({type(hw.strategy).__name__})"),
            ("precision", hw.precision), ("steps/call", str(hw.steps_per_call)),
            ("XLA (jit)", "auto: " + ("on" if hw.kind == "gpu" and hw.replicas == 1 and o["system"] != "macos" else "off")),
            ("TensorFlow", tf.__version__)]
    why = [f"why: {r}" for r in hw.reasons] + [f"note: {n}" for n in hw.notes]
    w = max(len(k) for k, _ in rows) + 2
    body = [f"  {k:<{w}}{v}" for k, v in rows] + [f"  {x}" for x in why]
    width = max(len(x) for x in body) + 2
    line = "+" + "-" * width + "+"
    title = " ApexNet hardware "
    top = "+" + title.center(width, "-") + "+"
    return "\n".join([top] + ["|" + x.ljust(width) + "|" for x in body] + [line])


def diagnose():
    """Print everything useful when setup() fails (paste this when asking for help)."""
    print("platform:", platform_name(), "| python", sys.version.split()[0], "| TF", tf.__version__)
    print("physical devices:", {k: tf.config.list_physical_devices(k) for k in ("CPU", "GPU", "TPU")})
    try:
        bi = tf.sysconfig.get_build_info()
        print("build:", {k: bi.get(k) for k in ("cuda_version", "cudnn_version", "is_cuda_build", "is_tpu_build")})
    except Exception as e:  # noqa
        print("build info unavailable:", e)
    env = {k: v for k, v in os.environ.items() if any(s in k for s in ("TPU", "COLAB", "KAGGLE", "CUDA", "TF_"))}
    print("env hints:", json.dumps(env, indent=1)[:1200])
    print("/dev/accel*:", glob.glob("/dev/accel*"))
