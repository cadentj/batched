from __future__ import annotations
import os

def warn_if_mps_daemon_inactive(
    cuda_mps_pipe_directory: str | None,
) -> None:
    pipe_directory = (
        cuda_mps_pipe_directory
        or os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
        or "/tmp/nvidia-mps"
    )
    control_socket = os.path.join(
        pipe_directory,
        "nvidia-cuda-mps-control",
    )
    if os.path.exists(control_socket):
        return

    print(
        "Warning: CUDA MPS daemon appears inactive "
        f"(missing {control_socket}). "
        "Start it with `nvidia-cuda-mps-control -d`."
    )