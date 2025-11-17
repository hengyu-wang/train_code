import torch
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainingArguments
from datasets import load_from_disk, Dataset, DatasetDict

# 导入你的自定义和 LLaMA-Factory 的模块
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras import logging
from llamafactory.hparams import DataArguments, ModelArguments, FinetuningArguments, GeneratingArguments, get_train_args
from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras import logging
# 我们不再使用 llamafactory 的通用加载器
from llamafactory.model.loader import load_model, load_tokenizer 
from custom_run_sft import custom_run_sft

logger = logging.get_logger(__name__)

# ==============================================================================
# --- 数据集加载函数 (已修改以适应 VLM) ---
# ==============================================================================
def load_preprocessed_vlm_dataset(
    dataset_path: str,
    tokenizer: "AutoTokenizer", # Tokenizer 仍然需要，用于添加特殊 Token
    model: "Qwen2_5_VLForConditionalGeneration", # Model 也需要，用于 resize_token_embeddings
    add_special_tokens: bool = True,
) -> Dict[str, "Dataset"]:
    """
    加载一个已经预处理好的 VLM 数据集，并处理特殊 Token。
    """
    logger.info(f"Loading preprocessed VLM dataset from {dataset_path}")
    
    try:
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        logger.error(f"Error loading dataset from disk: {e}")
        raise
        
    if add_special_tokens:
        special_tokens = [f"<think{i}>" for i in range(1, 9)] + [f"</think{i}>" for i in range(1, 9)] +["<summary>", "</summary>", "<vllm_pad>"]
        
        tokens_to_add = [tok for tok in special_tokens if tokenizer.convert_tokens_to_ids(tok) == tokenizer.unk_token_id]
                
        if tokens_to_add:
            logger.info(f"Adding {len(tokens_to_add)} special tokens to the tokenizer")
            tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
            logger.info("Resizing model token embeddings...")
            model.resize_token_embeddings(len(tokenizer))
            logger.info(f"Model token embeddings resized to {len(tokenizer)}")
            
    # Process the dataset structure (这部分逻辑保持不变)
    if isinstance(dataset, Dataset):
        dataset_dict = {"train_dataset": dataset}
    else:
        dataset_dict = {
            "train_dataset": dataset.get("train"),
            "eval_dataset": dataset.get("validation")
        }
    
    if dataset_dict["train_dataset"]:
        logger.info(f"Loaded train_dataset with {len(dataset_dict['train_dataset'])} rows.")
        logger.info(f"Features: {dataset_dict['train_dataset'].features}")
            
    return dataset_dict



def main():
    # 1. 解析所有参数
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()
    model_args.resize_vocab = True
    finetuning_args.pure_bf16 = True
    finetuning_args.plot_loss = True
    
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module['tokenizer']
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    # 2. ✅ 加载 VLM 特定的模型和 Processor
    logger.info(f"Loading VLM model and processor from {model_args.model_name_or_path}")
    # 直接使用 Auto* 类加载，绕过 LLaMA-Factory 的通用加载器
    # processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     model_args.model_name_or_path,
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16 # 使用 bfloat16 以节省内存
    # )
    # 从 processor 中提取 tokenizer 以便后续使用
    # tokenizer = processor.tokenizer
    
    # 3. ✅ 调用新的加载函数
    # 它会加载数据，并利用我们刚创建的 model 和 tokenizer 来添加特殊 Token
    dataset = load_preprocessed_vlm_dataset(
        dataset_path=data_args.dataset[0],
        tokenizer=tokenizer,
        model=model,
        add_special_tokens=True,
    )
    
    logger.info("Creating VLM-compatible data collator...")
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template = template,
        model=model, # 你的 collator 可能需要 model
        padding="longest",
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        # 你的 collator 的其他参数
        block_diag_attn=True, 
        compute_dtype=torch.bfloat16,
        **tokenizer_module
    )
    
    # 5. 运行 SFT 训练
    logger.info("Starting Supervised Fine-Tuning (SFT) run for VLM...")
    custom_run_sft(
        model_args=model_args, 
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=None,
        model=model,
        tokenizer=tokenizer,
        dataset_dict=dataset,
        data_collator=data_collator
    )
    
if __name__ == "__main__":
    main()
