# import json
# from datasets import Dataset
# from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration # 需要加载模型来调整嵌入层
# from PIL import Image
# import os
# import pprint

# # --- 1. 配置路径和数据 ---
# MODEL_PATH = "/mnt/bn/smart-customer-service/users/wanghongyu/ParaThinker/train-qwen2.5vl/LLaMA-Factory/model/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
# raw_data = [    
#     {
#         "messages": [
#             {"role": "user", "content": "<image>描述里面的内容"},
#             {"role": "assistant", "content": "<think1>图像中央有一只猫</think1><think2>猫是一只黑色的</think2><summary>图中是一只猫</summary>"}
#         ],
#         "images": ["/mnt/bn/smart-customer-service/users/wanghongyu/ParaThinker/train-qwen2.5vl/cat_example.jpg"]
#     },
#     {
#         "messages": [
#             {"role": "user", "content": "<image>描述里面的内容"},
#             {"role": "assistant", "content": "<think1>图像中央有一只猫</think1><think2>猫是一只黑色的</think2><summary>图中是一只猫</summary>"}
#         ],
#         "images": ["/mnt/bn/smart-customer-service/users/wanghongyu/ParaThinker/train-qwen2.5vl/cat_example.jpg"]
#     },
#     {
#         "messages": [
#             {"role": "user", "content": "<image>描述里面的内容"},
#             {"role": "assistant", "content": "<think1>图像中央有一只猫</think1><think2>猫是一只黑色的</think2><summary>图中是一只猫</summary>"}
#         ],
#         "images": ["/mnt/bn/smart-customer-service/users/wanghongyu/ParaThinker/train-qwen2.5vl/cat_example.jpg"]
#     },
#     {
#         "messages": [
#             {"role": "user", "content": "<image>描述里面的内容"},
#             {"role": "assistant", "content": "<think1>图像中央有一只猫</think1><think2>猫是一只黑色的</think2><summary>图中是一只猫</summary>"}
#         ],
#         "images": ["/mnt/bn/smart-customer-service/users/wanghongyu/ParaThinker/train-qwen2.5vl/cat_example.jpg"]
#     },{
#         "messages": [
#             {"role": "user", "content": "<image>描述里面的内容"},
#             {"role": "assistant", "content": "<think1>图像中央有一只猫</think1><think2>猫是一只黑色的</think2><summary>图中是一只猫</summary>"}
#         ],
#         "images": ["/mnt/bn/smart-customer-service/users/wanghongyu/ParaThinker/train-qwen2.5vl/cat_example.jpg"]
#     },
#     {
#         "messages": [
#             {"role": "user", "content": "<image>描述里面的内容"},
#             {"role": "assistant", "content": "<think1>图像中央有一只猫</think1><think2>猫是一只黑色的</think2><summary>图中是一只猫</summary>"}
#         ],
#         "images": ["/mnt/bn/smart-customer-service/users/wanghongyu/ParaThinker/train-qwen2.5vl/cat_example.jpg"]
#     }
# ]

# # --- 2. 加载 Processor 和 Model ---
# print("正在加载 Processor 和 Model...")
# processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     MODEL_PATH, 
#     trust_remote_code=True
# )

# # --- ✅ 3. 关键步骤：添加特殊 Token 并调整模型 ---
# # 定义所有你需要的特殊 Token
# special_tokens_list = [
#             "<think1>", "</think1>", "<think2>", "</think2>", "<think3>", "</think3>",
#             "<think4>", "</think4>", "<think5>", "</think5>", "<think6>", "</think6>",
#             "<think7>", "</think7>", "<think8>", "</think8>", "<summary>", "</summary>",
#             "<vllm_pad>"
#         ]

# print(f"\n准备添加的特殊 Token: {special_tokens_list}")

# # 使用 add_special_tokens 添加，它会自动处理重复项
# num_added_toks = processor.tokenizer.add_special_tokens({
#     "additional_special_tokens": special_tokens_list
# })

# if num_added_toks > 0:
#     print(f"成功添加了 {num_added_toks} 个新的特殊 Token。")
#     # 调整模型词嵌入层的大小以匹配新的 Tokenizer 词汇表大小
#     model.resize_token_embeddings(len(processor.tokenizer))
#     print("已成功调整模型嵌入层大小。")
# else:
#     print("所有特殊 Token 已存在，无需添加。")

# # --- 验证一下 ---
# print("\n--- 验证特殊 Token 是否被正确识别 ---")
# test_text = "<think1>这是一个测试</think1>"
# encoded_ids = processor.tokenizer.encode(test_text, add_special_tokens=False)
# print(f"'{test_text}' -> 编码后的 ID: {encoded_ids}")
# # 预期结果：<think1> 和 </think1> 都应该是一个单一的、高数值的 ID
# # 例如: [151683, 8948, 13, 123, 151684] 而不是 [8, 93, 1, 94, ...]

# decoded_text = processor.tokenizer.decode(encoded_ids)
# print(f"解码后的文本: '{decoded_text}'")
# print("------------------------------------")

# # --- 4. 后续的数据处理流程 (和之前一样) ---
# hf_dataset = Dataset.from_list(raw_data)

# def preprocess_function(example):
#     prompt = processor.apply_chat_template(
#         example["messages"], tokenize=False, add_generation_prompt=True
#     )
#     imgs = [Image.open(p).convert("RGB") for p in example["images"]]
#     # 1) 这里 processor 会把单条 sample 当 batch size=1 来处理，
#     #    所以所有返回的字段都是长度为1的 list
#     out = processor(
#         text=prompt,
#         images=imgs,
#         padding=False,
#         truncation=True,
#         max_length=2048,
#     )
#     print(out["image_grid_thw"])
#     # 2) 拆掉多余的第 0 维
#     return {
#         "input_ids":        out["input_ids"][0],
#         "attention_mask":   out["attention_mask"][0],
#         "pixel_values":     out["pixel_values"][0],
#         "image_grid_thw":   out["image_grid_thw"][0],
#     }

# print("\n正在应用预处理函数 (使用已更新的 Processor)...")
# processed_dataset = hf_dataset.map(
#     preprocess_function,
#     batched=False,
#     remove_columns=hf_dataset.column_names,
#     desc="正在转换数据..."
# )
# # --- 5. 查看最终结果 ---
# print("\n" + "="*80)
# print("转换完成！最终的数据集信息如下：")
# print(processed_dataset)
# print("="*80)
# for i, sample in enumerate(processed_dataset):
#     print(f"sample {i} input_ids = {sample['input_ids']}")
#     print(f"sample {i} pixel_values = {len(sample['pixel_values'])}")
#     print(f"sample {i} image_grid_thw = {sample['image_grid_thw']}")
# SAVE_PATH = "./data"
# print("\n" + "="*80)
# print(f"--- 正在将处理后的数据集保存到: {SAVE_PATH} ---")
# # 调用 .save_to_disk() 方法
# processed_dataset.save_to_disk(SAVE_PATH)
# print("数据集已成功保存！")
# print("="*80)


# print("\n" + "="*50)
# print("--- 正在验证所有特殊 Token 的 ID ---")
# # 使用 convert_tokens_to_ids 方法获取所有 ID
# special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens_list)
# # 将 Token 和 ID 一一对应地打印出来，方便查看
# special_token_map = dict(zip(special_tokens_list, special_token_ids))
# print("特殊 Token 到 ID 的映射关系如下：")
# pprint.pprint(special_token_map)
# # 也可以打印一下词汇表的大小，以作参考
# print(f"\n当前 Tokenizer 的词汇表大小为: {len(processor.tokenizer)}")
# print("="*50)
import json
from datasets import Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import os
import pprint

# --- 1. 配置路径和数据 ---
MODEL_PATH = "/mnt/bn/smart-customer-service/users/wanghongyu/ParaThinker/train-qwen2.5vl/LLaMA-Factory/model/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
raw_data = [    
    {
        "messages": [
            {"role": "user", "content": "<image>描述里面的内容"},
            {"role": "assistant", "content": "<think1>图像中央有一只猫</think1><think2>猫是一只黑色的</think2><summary>图中是一只猫</summary>"}
        ],
        "images": ["/mnt/bn/smart-customer-service/users/wanghongyu/ParaThinker/train-qwen2.5vl/cat_example.jpg"]
    },
    {
        "messages": [
            {"role": "user", "content": "<image>描述里面的内容"},
            {"role": "assistant", "content": "<think1>图像中央有一只猫</think1><think2>猫是一只黑色的</think2><summary>图中是一只猫</summary>"}
        ],
        "images": ["/mnt/bn/smart-customer-service/users/wanghongyu/ParaThinker/train-qwen2.5vl/cat_example.jpg"]
    },
    {
        "messages": [
            {"role": "user", "content": "<image>描述里面的内容"},
            {"role": "assistant", "content": "<think1>图像中央有一只猫</think1><think2>猫是一只黑色的</think2><summary>图中是一只猫</summary>"}
        ],
        "images": ["/mnt/bn/smart-customer-service/users/wanghongyu/ParaThinker/train-qwen2.5vl/cat_example.jpg"]
    }
    # 为了演示，我们只用一个样本，方便查看输出
    # 你可以随时把其他样本加回来
]

# --- 2. 加载 Processor 和 Model ---
print("正在加载 Processor 和 Model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True
)

# --- 3. 添加特殊 Token 并调整模型 ---
special_tokens_list = [
    "<think1>", "</think1>", "<think2>", "</think2>", "<think3>", "</think3>",
    "<think4>", "</think4>", "<think5>", "</think5>", "<think6>", "</think6>",
    "<think7>", "</think7>", "<think8>", "</think8>", "<summary>", "</summary>",
    "<vllm_pad>"
]

print(f"\n准备添加的特殊 Token: {special_tokens_list}")
num_added_toks = processor.tokenizer.add_special_tokens({
    "additional_special_tokens": special_tokens_list
})

if num_added_toks > 0:
    print(f"成功添加了 {num_added_toks} 个新的特殊 Token。")
    model.resize_token_embeddings(len(processor.tokenizer))
    print("已成功调整模型嵌入层大小。")
else:
    print("所有特殊 Token 已存在，无需添加。")

# --- 4. 修改后的数据处理函数 ---
# 这是核心修改部分
def preprocess_function_with_labels(example):
    """
    该函数将原始样本转换为SFT所需的格式，包含正确的 `labels` 字段。
    """
    IGNORE_INDEX = -100
    
    # 1. 分离 prompt 和 completion
    messages = example["messages"]
    prompt_messages = messages[:-1]  # 除了最后一个助手消息外的所有内容
    completion_messages = messages[-1:] # 只有最后一个助手消息

    # 2. 应用聊天模板
    # a) 获取 prompt 部分的文本，并附加上让模型开始生成回答的指令
    prompt_part = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    # b) 获取 completion 部分的文本，并加上结束符 (EOS token)
    completion_part = processor.apply_chat_template(
        completion_messages, tokenize=False, add_generation_prompt=False
    ) + processor.tokenizer.eos_token

    # 3. 单独对 prompt 部分进行分词，以确定需要 mask 的长度
    prompt_ids = processor.tokenizer(prompt_part, add_special_tokens=False).input_ids
    prompt_length = len(prompt_ids)

    # 4. 对完整的对话进行分词，得到最终的 input_ids
    full_text = prompt_part + completion_part
    
    # 注意：这里的 processor 调用不会处理图像，因为它只接收 text 参数。
    # 这是一个纯文本的 tokenization 过程，符合 Qwen2.5-VL 的处理方式，
    # 即文本和图像是分开处理然后由模型内部融合的。
    # 你的 DataCollator 会负责处理图像并生成 pixel_values。
    input_ids = processor.tokenizer(full_text, add_special_tokens=False).input_ids

    # 5. 创建 labels 数组
    # 将 prompt 部分的标签设为 IGNORE_INDEX
    labels = [IGNORE_INDEX] * prompt_length + input_ids[prompt_length:]

    # 6. 确保长度一致性
    if len(input_ids) != len(labels):
        raise ValueError("Input IDs 和 Labels 的长度不匹配！")

    # 7. 返回目标格式的字典
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "images": example["images"],  # 保留原始图像路径
        "videos": None,               # 按要求添加
        "audios": None,               # 按要求添加
    }

# --- 5. 应用新的预处理函数 ---
hf_dataset = Dataset.from_list(raw_data)

print("\n正在应用新的预处理函数 (以生成 SFT 格式)...")
processed_dataset = hf_dataset.map(
    preprocess_function_with_labels, # <-- 使用新的函数
    batched=False,
    remove_columns=hf_dataset.column_names,
    desc="正在转换数据..."
)

# --- 6. 查看并验证最终结果 ---
print("\n" + "="*80)
print("转换完成！最终的数据集信息如下：")
print(processed_dataset)
print("="*80)

# 打印第一个样本以进行详细检查
final_sample = processed_dataset[0]
print("\n--- 生成的第一个样本（目标格式）---")
# 为了美观，我们只打印部分长列表
print(f"'input_ids': (len={len(final_sample['input_ids'])})")
print(f"   {final_sample['input_ids']}...")
print(f"'attention_mask': (len={len(final_sample['attention_mask'])})")
print(f"   {final_sample['attention_mask']}...")
print(f"'labels': (len={len(final_sample['labels'])})")
print(f"   {final_sample['labels']}...") # 应该以 -100 开头
# 找到第一个非-100的标签，验证其与input_ids是否匹配
first_label_idx = -1
for i, label in enumerate(final_sample['labels']):
    if label != -100:
        first_label_idx = i
        break
print(f"\nPrompt 部分被 mask 的长度: {first_label_idx}")
if first_label_idx != -1:
    print(f"第一个有效 label: {final_sample['labels'][first_label_idx]}, 对应的 input_id: {final_sample['input_ids'][first_label_idx]} (应相同)")
    print(f"最后一个有效 label: {final_sample['labels'][-2]}, 对应的 input_id: {final_sample['input_ids'][-2]} (应相同, -1是EOS)")

print(f"\n'images': {final_sample['images']}")
print(f"'videos': {final_sample['videos']}")
print(f"'audios': {final_sample['audios']}")
print("="*80)

# --- 7. 保存处理后的数据集 ---
SAVE_PATH = "./data_sft_format"
print(f"\n--- 正在将处理后的数据集保存到: {SAVE_PATH} ---")
processed_dataset.save_to_disk(SAVE_PATH)
print("数据集已成功保存！")
print("="*80)
