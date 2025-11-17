export DISABLE_VERSION_CHECK=1

# 使用 torchrun 来启动脚本
# --nproc_per_node 的值应该等于您想使用的GPU数量
torchrun --nproc_per_node=4 qwen2_5vl.py \
    --stage sft \
    --do_train \
    --model_name_or_path /mnt/bn/smart-customer-service/users/wanghongyu/ParaThinker/train-qwen2.5vl/LLaMA-Factory/model/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
    --dataset /mnt/bn/smart-customer-service/users/wanghongyu/ParaThinker/train-qwen2.5vl/data_sft_format \
    --output_dir ./output_parallel_think \
    --template qwen2_vl \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --bf16 \
    --deepspeed /mnt/bn/smart-customer-service/users/wanghongyu/ParaThinker/train-qwen2.5vl/LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
    --max_length 1024 \
    --finetuning_type full