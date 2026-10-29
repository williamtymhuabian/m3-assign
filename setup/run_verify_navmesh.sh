#!/bin/bash
# habitat-sim 0.1.7 跑在 m3 环境；EGL 无头渲染需要指定 NVIDIA ICD 与驱动 GL 库目录。
cd "$(dirname "$0")/.." || exit 1
export EGL_PLATFORM=surfaceless
export __EGL_VENDOR_LIBRARY_FILENAMES="$PWD/setup/10_nvidia.json"
export LD_LIBRARY_PATH="${NVIDIA_GL_LIB:-$HOME/nvidia-gl/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "${M3_PYTHON:-$HOME/anaconda3/envs/m3/bin/python}" -m m3assign.verify_navmesh "${1:-8}"
