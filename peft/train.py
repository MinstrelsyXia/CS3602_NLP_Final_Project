from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    HfArgumentParser,
    Trainer,
)
import torch
from dataclasses import dataclass, field
import sys
import wandb
import argparse
import os
from typing import Optional, List, Dict


import datasets
# 配置参数
@dataclass
class ModelArguments:
    """Arguments for model
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the LLM to fine-tune or its name on the Hugging Face Hub."
        }
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype."
            ),
            "choices": ["bfloat16", "float16", "float32"],
        },
    )
    # TODO: add your model arguments here
    pass
@dataclass
class DataArguments:
    """Arguments for data
    """
    dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the fine-tuning dataset or its name on the Hugging Face Hub."
        }
    )
    # TODO: add your data arguments here
    max_length: int = field(
        default=512,
        metadata={
            "help": "The max length of tokenized data."
        }
    )
    skip_too_long: Optional[bool] = field(
        default = False,
        metadata = {
            "help" : "whether to skip those longer than max length of tokenized data"
        }
    )

def finetune(peft_config):
    parser = HfArgumentParser(dataclass_types=[ModelArguments, DataArguments, TrainingArguments])
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,  # Qwen模型需要这个参数
        device_map="balanced" 
    )


    def data_collator(batch: List[Dict]):
        """
        batch: list of dict, each dict of the list is a sample in the dataset.
        """
        # pass
        ### cjy
        ## cyd
        inputs = []
        labels = []
        max_length = 0

        for sample in batch:
            instruction = sample.get("instruction","")
            input_text = sample.get("input","")
            output_text = sample.get("output","")

            # # 检查input和output是否为空
            # if not input_text.strip():
            #     input_text = f"\n{input_text}"

            input_text_special = f"instruction: {instruction}\ninput: {input_text}"
            output_text_special = output_text
            # input_text_special, output_text_special = generate_prompt(sample)
            # 构建输入序列
            input_ids = tokenizer.encode_plus(
                input_text_special, # 将instruction和input_text进行拼接，生成文本输入
                return_tensors = "pt", # 输出转换为pytorch的张量格式
                max_length = tokenizer.model_max_length, # 如果输入序列超过最大长度则截断
                truncation = True, 
                padding = False, # 即使输入序列没有达到最大长度，也不进行填充
            ).input_ids # 用于获取tokenizer返回字典中的‘input_ids’字段
            # if input_ids[0, -1] == tokenizer.eos_token_id:
            #     input_ids = input_ids[:, :-1]
            # # 构建输出序列
            output_ids = tokenizer.encode_plus(
                output_text_special,
                return_tensors = "pt",
                max_length = tokenizer.model_max_length,
                truncation = True,
                padding = False,
            ).input_ids
            if input_ids[0, -1] != tokenizer.eos_token_id:
                input_ids = torch.cat([input_ids, torch.tensor([[tokenizer.eos_token_id]], device=input_ids.device)], dim=1)
            # 如果output_ids的最后一个token不是eos_token_id，则添加一个eos_token_id
            if output_ids[0, -1] != tokenizer.eos_token_id:
                output_ids = torch.cat([output_ids, torch.tensor([[tokenizer.eos_token_id]], device=output_ids.device)], dim=1)
            # 拼接输入与输出序列，获得模型所需的input_ids
            full_input = torch.cat([input_ids, output_ids], dim=1)
            # if (full_input.size(1)>data_args.max_length):
            #     continue
            inputs.append(full_input)

            # 创建标签张量
            labels_tensor = torch.full_like(full_input, -100) # 用-100填充表示这些位置在损失计算中被忽略
            labels_tensor[:, input_ids.shape[1]:] = full_input[:, input_ids.shape[1]:] # 将output_ids对应的位置替换为output_ids原来的值，input_ids对应的位置仍为-100，表示学习时只学习output部分
            labels.append(labels_tensor)

            if full_input.shape[1] > max_length:
                max_length = full_input.shape[1] + 5


        # 处理batch的padding，将同一个batch的label和input都填充到同一个长度
        # inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
        # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        inputs = [torch.nn.functional.pad(input, (0,max_length - input.size(1)),value=tokenizer.pad_token_id) for input in inputs]
        labels = [torch.nn.functional.pad(label, (0, max_length - label.size(1)), value=-100) for label in labels]

        # inputs = torch.stack(inputs)
        # labels = torch.stack(labels)

        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)

        return {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": ((inputs != tokenizer.pad_token_id)).to(dtype=torch.int), 
            # pad_token_id是tokenizer定义的填充令牌ID，也就是对padding的部分填充一个特殊的令牌
            # 该行代码将生成一个与inputs张量形状相同的布尔张量，其中值为True:表示对应的输入ID不是填充ID（即该令牌是有效的；值为False: 表示对应的输入ID是填充ID（即该令牌是无效的，需要被忽略）。
            # ((inputs != tokenizer.pad_token_id) & (inputs != labels)).to(dtype=torch.int)
        }
    # 获取 PEFT 模型
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 加载数据集（使用与 finetune.py 相同的数据处理逻辑）

    dataset = datasets.load_dataset(path='csv', data_files=data_args.dataset_path)

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset['train'],
        tokenizer=tokenizer,
    )
    # 开始训练
    trainer.train()

def main():

    parser = argparse.ArgumentParser(description="Fine-tune a model.")

    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,5,6,7', help='Comma-separated list of GPU IDs to use')
    # parser.add_argument('--version', type=str, default='0.5B', help='Version of the model')
    # parser.add_argument('--note', type=str, default='', help='Note for the model')
    args = parser.parse_args()
    # 创建配置字典
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    config_dict = {
        "model_name_or_path": "/home/xiaxinyuan/.cache/kagglehub/models/qwen-lm/qwen2.5/transformers/3b/1/",
        "dataset_path": "/home/xiaxinyuan/.cache/kagglehub/datasets/thedevastator/alpaca-language-instruction-training/versions/2/train.csv",
        "torch_dtype": "float32",
        "output_dir": "/ssd/xiaxinyuan/code/CS3602_NLP_Final_Project/output/peft_output/",
        "remove_unused_columns": "False",
        "max_length": "512",
        "skip_too_long": "True",
        "learning_rate": "5e-5",
        "lr_scheduler_type": "cosine",
        "optim": "adamw_hf",
        "warmup_ratio": "0.03",
        "weight_decay": "0.003",
        "per_device_train_batch_size": "8",
        "per_device_eval_batch_size": "4",
        "save_steps": "10000",
        "save_total_limit": "4",
        "num_train_epochs": "5",
        "bf16": "True",
        "fp16": "False",
        "overwrite_output_dir": "True",
        "seed": "42",
    }
    lora_config = {
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }    
    # 更新 sys.argv
    sys.argv = [
            "notebook",
            "--model_name_or_path", config_dict["model_name_or_path"],
            "--dataset_path", config_dict["dataset_path"],
            "--torch_dtype", config_dict["torch_dtype"],
            "--output_dir", config_dict["output_dir"],
            "--remove_unused_columns", config_dict["remove_unused_columns"],
            "--max_length", config_dict["max_length"],
            "--skip_too_long", config_dict["skip_too_long"],
            "--learning_rate", config_dict["learning_rate"],
            "--lr_scheduler_type", config_dict["lr_scheduler_type"],
            "--optim", config_dict["optim"],
            "--warmup_ratio", config_dict["warmup_ratio"],
            "--weight_decay", config_dict["weight_decay"],
            "--per_device_train_batch_size", config_dict["per_device_train_batch_size"],
            "--per_device_eval_batch_size", config_dict["per_device_eval_batch_size"],
            "--save_steps", config_dict["save_steps"],
            "--save_total_limit", config_dict["save_total_limit"],
            "--num_train_epochs", config_dict["num_train_epochs"],
            "--fp16", config_dict["fp16"],
            "--bf16", config_dict["bf16"],
            "--overwrite_output_dir", config_dict["overwrite_output_dir"],
            "--seed", config_dict["seed"],
        ]
    
    
    wandb_dict = config_dict.copy() | lora_config.copy()
    # wandb.init(mode='disabled')
    wandb.login(key="5e4de12fa847ce69f658bd4cd6ef1819aa110ed5")
    # 使用config_dict而不是sys.argv
    wandb.init(project="CS3602_NLP_Final_Project", config=wandb_dict, tags=["Qwen2.5-3B"])
    
    # 配置 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 使用 CAUSAL_LM 而不是 SEQ_2_SEQ_LM
        inference_mode=False,
        r=lora_config["r"]  ,  # LoRA 秩
        lora_alpha=lora_config["lora_alpha"],  # LoRA alpha参数
        lora_dropout=lora_config["lora_dropout"],
        # 可以添加更多配置
        target_modules=lora_config["target_modules"],  # 指定需要训练的模块
    )

    finetune(peft_config)

    # # 保存模型
    # trainer.save_model(output_dir=config_dict["output_dir"])

if __name__ == "__main__":
    main()