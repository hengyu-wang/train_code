#!/bin/bash

# 脚本开头使用 set -e，确保任何命令失败时，脚本都会立即退出
set -e

# --- 1. 定义路径 ---
# 获取当前所在的网络驱动器上的项目根目录
# realpath 将处理 . 和 ..，得到一个绝对路径
SOURCE_DIR=$(realpath .)
# 在本地磁盘 /tmp 下创建一个唯一的、临时的构建目录
# 使用项目文件夹的名字作为前缀，避免冲突
BUILD_DIR="/tmp/$(basename "${SOURCE_DIR}")_build_$(date +%s)"

echo "-----------------------------------------------------"
echo "Source project directory (on network drive): ${SOURCE_DIR}"
echo "Temporary build directory (on local disk):   ${BUILD_DIR}"
echo "-----------------------------------------------------"
echo

# --- 2. 将项目代码复制到本地磁盘 ---
echo "Step 1: Copying project files to local disk for compilation..."
# 使用 rsync -a 可以高效地复制文件并保持权限等属性
rsync -a --delete "${SOURCE_DIR}/" "${BUILD_DIR}/"
echo "✅ Project copied successfully."
echo

# --- 3. 进入本地构建目录并开始安装 ---
# 从现在开始，所有操作都在本地磁盘上进行，不会再有权限问题
cd "${BUILD_DIR}"

echo "Step 2: Creating a clean virtual environment in the local build directory..."
# (如果 uv 未安装，这行会安装它)
curl -LsSf https://astral.sh/uv/install.sh | sh
# 在本地目录创建 .venv
UV_VENV_CLEAR=1 uv venv --python 3.11
echo "✅ Virtual environment created."
echo

echo "Step 3: Activating the virtual environment..."
source .venv/bin/activate
echo "✅ Environment activated."
# 检查一下，确保 uv 和 python 都来自本地的 .venv
echo "Using uv from: $(which uv)"
echo "Using python from: $(which python)"
echo

echo "Step 4: Installing LLaMA-Factory in editable mode..."
cd ./LLaMA-Factory
# 注意：使用 uv pip，而不是 uv pip3
uv pip install --editable ".[torch,metrics]"
cd ..
echo "✅ LLaMA-Factory installed."
echo

echo "Step 5: Installing local transformers library in editable mode..."
cd ./transformers
uv pip install --editable .
cd ..
echo "✅ Local transformers installed."
echo

# --- 4. 完成 ---
echo "🎉🎉🎉 Environment setup complete! 🎉🎉🎉"
echo
echo "You are now in the temporary build directory: ${BUILD_DIR}"
echo "The virtual environment is active. You can start your development and training here."
echo
echo "IMPORTANT:"
echo "1. All your work (code changes, training) should be done in THIS directory."
echo "2. When you are finished, copy your results (logs, checkpoints) back to '${SOURCE_DIR}'."
echo "3. This temporary directory in /tmp might be deleted upon system reboot."
echo
echo "To enter this environment again later (before a reboot), run:"
echo "cd ${BUILD_DIR} && source .venv/bin/activate"
echo
cd /tmp/train-qwen2.5vl_build_1763115447
source .venv/bin/activate
cd /mnt/bn/smart-customer-service/users/wanghongyu/ParaThinker/train-qwen2.5vl
# 保持 shell 打开，让你直接在这个新环境中工作
# 如果你希望脚本运行后自动退出，可以注释掉下面这行
exec bash