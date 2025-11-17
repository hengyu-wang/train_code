echo "=================================================="
echo "开始执行环境清理操作..."
echo "目标目录: $(pwd)"
echo "=================================================="
# 1. 删除 .venv 虚拟环境文件夹
echo "正在删除 .venv 文件夹..."
rm -rf .venv
echo ".venv 已删除。"
# 2. 删除 uv 的项目配置文件
echo "正在删除 pyproject.toml 和 uv.lock..."
rm -f pyproject.toml uv.lock
echo "项目配置文件已删除。"
# 3. 查找并删除所有 __pycache__ 文件夹
echo "正在清理所有 __pycache__ 文件夹..."
find . -type d -name "__pycache__" -exec rm -rf {} +
echo "__pycache__ 已清理。"
echo "清理完成！"