"""Select the ML backend once, before importing pandas or scikit-learn."""

import platform
import subprocess
from importlib.util import find_spec


def _has_spark_gpu() -> bool:
    if platform.system() != "Linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
        return False

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    return "GB10" in result.stdout.upper()


def _configure_backend() -> tuple[bool, str]:
    if not _has_spark_gpu():
        return False, "CPU"

    if any(find_spec(name) is None for name in ("cudf", "cuml", "cupy")):
        return False, "CPU · RAPIDS packages unavailable"

    try:
        from cupy.cuda.runtime import getDeviceCount

        if getDeviceCount() == 0:
            return False, "CPU · CUDA device unavailable"
    except (ImportError, OSError, RuntimeError):
        return False, "CPU · CUDA runtime unavailable"

    # Install cuDF first so any subsequent pandas imports see its proxy.
    from cudf import pandas as cudf_pandas

    cudf_pandas.install()

    from cuml import accel

    accel.install()
    return True, "GPU · RAPIDS on GB10"


GPU_ENABLED, ML_BACKEND = _configure_backend()
print(f"GPU_ENABLED: {GPU_ENABLED}, ML_BACKEND: {ML_BACKEND}")