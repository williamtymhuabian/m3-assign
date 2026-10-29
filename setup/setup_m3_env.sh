#!/usr/bin/env bash
# 按 sim-code/environment.yml 原生搭建 habitat-sim 0.1.7 环境 m3（无 Docker）。
# 可重复执行：每一步先判断是否已完成。日志 /tmp/m3-env-setup.log
# 路径可用环境变量覆盖：M3_ROOT / M3_PREFIX（conda 环境目录）/ CONDA_BIN
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${M3_ROOT:-$(cd "$HERE/../.." && pwd)}"
SIM="$ROOT/hm3d-m3-code/sim-code"
PFX="${M3_PREFIX:-$HOME/anaconda3/envs/m3}"
CONDA="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
PY="$PFX/bin/python"
PIP="$PY -m pip"
DEPS="$HERE/deps"
LOG=/tmp/m3-env-setup.log
GX=https://gh-proxy.org/https://github.com          # 本机直连 github.com 不通
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
export PYTHONNOUSERSITE=1
HAB_SIM="habitat-sim=0.1.7=py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7"
HL_SHA=d6ed1c0a0e786f16f261de2beafe347f4186d0d8
GD_SHA=57535c5a79791cb76e36fdb64975271354f10251
SA_SHA=dca509fe793f601edb92606367a655c15ac00fdf

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() { say "\$ $*"; "$@" >>"$LOG" 2>&1 || { say "FAILED: $*"; return 1; }; }
mkdir -p "$DEPS"

step_conda() {
  [ -x "$PY" ] && { say "conda env 已存在，跳过创建"; return 0; }
  run "$CONDA" create -y -p "$PFX" -c conda-forge python=3.8.20 pip || return 1
}

step_habitat_sim() {
  "$PY" -c "import habitat_sim" >>"$LOG" 2>&1 && { say "habitat-sim 已装"; return 0; }
  run "$CONDA" install -y -p "$PFX" -c aihabitat -c conda-forge "$HAB_SIM" || return 1
  # conda-forge 的 libglvnd 会让 EGL 取不到 GL 版本，按 environment.yml 说明移除
  "$CONDA" remove -y -p "$PFX" --force libgl libegl libglx libglvnd >>"$LOG" 2>&1 || true
}

step_torch() {
  "$PY" -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.4.1') else 1)" >>"$LOG" 2>&1 \
    && { say "torch 2.4.1 已装"; return 0; }
  run $PIP install --no-cache-dir torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 \
      --index-url https://download.pytorch.org/whl/cu118 || return 1
}

step_reqs() {
  "$PY" -c "import imageio, quaternion" >>"$LOG" 2>&1 && { say "requirements 已装"; return 0; }
  run $PIP install --no-cache-dir "setuptools<70" wheel || return 1
  local req="$DEPS/requirements-no-git.txt"
  grep -v "^segment_anything @" "$SIM/requirements.txt" >"$req"
  run $PIP install --no-cache-dir -r "$req" || return 1
  local sa="$DEPS/segment-anything.tar.gz"
  [ -s "$sa" ] || run curl -sSL --retry 3 -o "$sa" "$GX/facebookresearch/segment-anything/archive/$SA_SHA.tar.gz" || return 1
  run $PIP install --no-cache-dir "$sa" || return 1
}

step_habitat_lab() {
  "$PY" -c "import habitat" >>"$LOG" 2>&1 && { say "habitat-lab 已装"; return 0; }
  local src="$DEPS/habitat-lab"
  if [ ! -d "$src" ]; then
    run curl -sSL --retry 3 -o "$DEPS/hl.tar.gz" "$GX/facebookresearch/habitat-lab/archive/$HL_SHA.tar.gz" || return 1
    run tar xzf "$DEPS/hl.tar.gz" -C "$DEPS" || return 1
    run mv "$DEPS/habitat-lab-$HL_SHA" "$src" || return 1
  fi
  if [ ! -e "$src/.vln-patched" ]; then
    ( cd "$src" && patch -p1 --forward < "$SIM/docker/patches/habitat-lab-vln.patch" ) >>"$LOG" 2>&1 \
      || { say "FAILED: habitat-lab patch"; return 1; }
    touch "$src/.vln-patched"
  fi
  # 该 commit 是扁平布局（habitat/ 与 habitat_baselines/ 在根目录），与 Dockerfile 一致
  run env PYTHONPATH= "$PY" "$src/setup.py" develop --all --no-deps || return 1
}

step_grounding_dino() {
  "$PY" -c "import groundingdino" >>"$LOG" 2>&1 && { say "GroundingDINO 已装"; return 0; }
  local src="$DEPS/GroundingDINO"
  if [ ! -d "$src" ]; then
    run curl -sSL --retry 3 -o "$DEPS/gd.tar.gz" "$GX/IDEA-Research/GroundingDINO/archive/$GD_SHA.tar.gz" || return 1
    run tar xzf "$DEPS/gd.tar.gz" -C "$DEPS" || return 1
    run mv "$DEPS/GroundingDINO-$GD_SHA" "$src" || return 1
  fi
  run env CUDA_HOME=/usr/local/cuda-11.8 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" \
      $PIP install --no-build-isolation --no-deps -e "$src" || return 1
}

for s in conda habitat_sim torch reqs habitat_lab grounding_dino; do
  say "=== step $s"
  "step_$s" || { say "中止于 $s（见 $LOG）"; exit 1; }
done

say "=== 验证"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import torch, habitat_sim, numpy
print('python', __import__('sys').version.split()[0])
print('torch', torch.__version__, 'cuda ok', torch.cuda.is_available())
print('habitat_sim', habitat_sim.__version__, habitat_sim.__file__)
print('numpy', numpy.__version__)
PY
say "完成"
