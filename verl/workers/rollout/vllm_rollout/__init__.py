import os
from importlib.metadata import PackageNotFoundError, version
from .vllm_rollout import vLLMAsyncRollout


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


vllm_package_name = "vllm"
vllm_package_version = get_version(vllm_package_name)
if vllm_package_version is None:
    raise PackageNotFoundError(
        "To use vllm rollout, please ensure the 'vllm' package is properly installed. See  for more details"
    )
if "ROCM_PATH" in os.environ:
    import re

    match = re.match("(\\d+\\.\\d+\\.?\\d*)", vllm_package_version)
    if match:
        vllm_package_version = match.group(1)
    else:
        raise ValueError(
            f"Warning: Could not parse version format: {vllm_package_version}"
        )
