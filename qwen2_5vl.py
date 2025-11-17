import os
import sys
from typing import Optional

# ------------------------------------------------------------------------------------
# 关键：确保可以导入 LLaMA-Factory 的模块。
# 如果你通过 `pip install -e .` 安装了 LLaMA-Factory，通常不需要手动添加路径。
# 否则，请取消下面的注释，并指向你的 LLaMA-Factory/src 目录。
#
# import sys
# sys.path.append("/path/to/your/LLaMA-Factory/src")
# ------------------------------------------------------------------------------------

from transformers import Seq2SeqTrainingArguments, TrainerCallback
from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps
from llamafactory.extras.ploting import plot_loss
from llamafactory.hparams import get_train_args, DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

logger = get_logger(__name__)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    """
    一个功能完整的、独立的 SFT 运行函数，它复制了 LLaMA-Factory 内部 run_sft 的所有逻辑。
    """
    # 阶段一：准备工作 (Setup and Loading)
    logger.info("阶段一：准备工作 (Setup and Loading)")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # 使用框架标准的 get_dataset，它会根据 data_args.dataset 自动查找和处理数据
    # 注意：你的命令中 `--dataset` 的值是一个路径，get_dataset 会自动处理这种情况
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)

    # 阶段二：配置核心组件 (Configuring Core Components)
    logger.info("阶段二：配置核心组件 (Configuring Core Components)")
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # 配置评估指标
    metric_module = {}
    if training_args.predict_with_generate:
        logger.info("使用生成模式进行评估，配置 ComputeSimilarity (ROUGE/BLEU)。")
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        logger.info("使用分类模式进行评估，配置 ComputeAccuracy。")
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # 配置生成参数
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # 阶段三：初始化训练器 (Initializing the Trainer)
    logger.info("阶段三：初始化训练器 (Initializing the Trainer)")
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # 阶段四：执行任务 (Execution)
    logger.info("阶段四：执行任务 (Execution)")
    if training_args.do_train:
        logger.info("--- 开始训练 ---")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        # trainer.save_model()
    #     if finetuning_args.include_effective_tokens_per_second:
    #         train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
    #             dataset_module["train_dataset"], train_result.metrics, stage="sft"
    #         )

    #     trainer.log_metrics("train", train_result.metrics)
    #     trainer.save_metrics("train", train_result.metrics)
    #     trainer.save_state()
    #     if trainer.is_world_process_zero() and finetuning_args.plot_loss:
    #         logger.info("绘制损失曲线...")
    #         plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    # if training_args.predict_with_generate:
    #     tokenizer.padding_side = "left"

    # if training_args.do_eval:
    #     logger.info("--- 开始评估 ---")
    #     metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # if training_args.do_predict:
    #     logger.info("--- 开始预测 ---")
    #     predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
    #     trainer.log_metrics("predict", predict_results.metrics)
    #     trainer.save_metrics("predict", predict_results.metrics)
    #     trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # # 阶段五：收尾工作 (Finalization)
    # logger.info("阶段五：收尾工作 (Finalization)")
    # create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)


def main():
    # 1. 解析所有命令行参数
    # get_train_args() 会自动读取 sys.argv 并解析所有你提供的参数
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()

    # 2. 直接运行 SFT 任务
    # 因为你是用 torchrun 启动的，所以脚本本身不需要再处理分布式启动的逻辑。
    # torchrun 已经为每个进程设置好了环境变量 (RANK, WORLD_SIZE 等)。
    # 我们直接进入核心训练逻辑即可。
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args)


if __name__ == "__main__":
    main()
