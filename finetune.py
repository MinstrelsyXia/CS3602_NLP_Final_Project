import wandb

import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import sys
import torch
from transformers import TrainingArguments, HfArgumentParser, Trainer, AutoTokenizer, AutoModelForCausalLM
import datasets
import argparse
import os
# The main function
# NOTE You can customize some logs to monitor your program.
# Define the arguments required for the main program.
# NOTE: You can customize any arguments you need to pass in.
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

def generate_prompt(data_point, is_logger=False):
    """ 直接encode，最后强制补上(不走新增了) """
    # sorry about the formatting disaster gotta move fast
    # text_1 = f"指令：\n{data_point.get('instruction', '')}\n问：\n{data_point.get('input', '')}\n答：\n" \
    #     if data_point.get('input', '') else f"指令：\n{data_point.get('instruction', '')}\n答：\n"
    # text_2 = f"{data_point.get('output', '')}"

    # text_input = data_point.get("instruction", "") + "\t" + data_point.get("input", "")
    # text_out = data_point.get("output", "")
    text_input = data_point.get("instruction", "") + "\t" + data_point.get("input", "")
    text_out = data_point.get("output", "")

    system_str = "You are a helpful assistant."
    prompt_system = "<|im_start|>system\n{}<|im_end|>\n".format(system_str)
    prompt_text_1 = prompt_system + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>"
    prompt_text_2 = "assistant\n{}<|im_end|><|endoftext|>"
    text_1 = prompt_text_1.format(text_input.strip())
    text_2 = prompt_text_2.format(text_out)
    # end with <|im_end|><|endoftext|>
    return text_1, text_2

def finetune():
    # TODO Step 1: Define an arguments parser and parse the arguments
    # NOTE Three parts: model arguments, data arguments, and training arguments
    # HINT: Refer to 
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/internal/trainer_utils#transformers.HfArgumentParser
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/trainer#transformers.TrainingArguments
    # parser = ...
    # model_args, data_args, training_args = ...
    ### cjy
    parser = HfArgumentParser(dataclass_types=[ModelArguments, DataArguments, TrainingArguments])
    """
    hugging face argument parser
    """
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    """
    return tuple
    DataClass 实例的顺序与传递给 initializer.abspath 的顺序相同
    如果适用，将更多（非 DataClass 支持的）参数的附加命名空间添加到解析器 初始化后。
    剩余参数字符串的可能列表。（与 argparse 相同。ArgumentParser.parse_known_args）
    """
    ### cjy

    # TODO Step 2: Load tokenizer and model
    # HINT 1: Refer to
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/tokenizer#tokenizer
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/qwen2
    # HINT 2: To save training GPU memory, you need to set the model's parameter precision to half-precision (float16 or bfloat16).
    #         You may also check other strategies to save the memory!
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/llama2#usage-tips
    #   * https://huggingface.co/docs/transformers/perf_train_gpu_one
    #   * https://www.53ai.com/news/qianyanjishu/2024052494875.html
    # tokenizer = ...
    # model = ...
    ## cjy
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_args.model_name_or_path)
    """
    从预训练模型的 tokenizer 配置中加载 tokenizer 的配置
    input:
        pretrained_model_name_or_path: 预训练模型的名称或路径（tokenizer 配置文件 所在的目录 即“Qwen2.5-0.5B”）
    功能:
        1 函数首先尝试从指定的模型名称或路径加载 tokenizer 配置文件
        2 如果找不到配置文件，则从模型库中加载默认的 tokenizer 配置
    """
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path, 
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=True,  # Qwen模型需要这个参数
        device_map="balanced"  # 可选，用于自动处理模型加载到设备
    )
    """
    用于从预训练模型加载模型实例。 
    """
    ### cjy

    # TODO Step 3: Load dataset
    # HINT: https://huggingface.co/docs/datasets/v3.1.0/en/package_reference/main_classes#datasets.Dataset
    # dataset = ...
    ### cjy
    dataset = datasets.load_dataset(path='csv', data_files=data_args.dataset_path)
    ### cjy

    # TODO Step 4: Define the data collator function
    # NOTE During training, for each model parameter update, we fetch a batch of data, perform a forward and backward pass,
    # and then update the model parameters. The role of the data collator is to process the data (e.g., padding the data within
    # a batch to the same length) and format the batch into the input required by the model.
    # 数据整理器（data collator）函数。这个函数的主要任务是处理从数据集中加载的每个数据批次，使其符合模型所需的输入格式
    # In this assignment, the purpose of the custom data_collator is to process each batch of data from the dataset loaded in
    # Step 3 into the format required by the model. This includes tasks such as tokenizing the data, converting each token into 
    # an ID sequence, applying padding, and preparing labels.
    # 对数据进行标记化（tokenization）。将每个标记转换为 ID 序列。填充（padding），确保批次中的所有数据具有相同的长度。准备标签（labels），以便在训练过程中使用
    # HINT:
    #   * Before implementation, you should:
    #      1. Clearly understand the format of each sample in the dataset loaded in Step 3.
    #      2. Understand the input format required by the model (https://huggingface.co/docs/transformers/model_doc/qwen2#transformers.Qwen2ForCausalLM).
    #         Reading its source code also helps!

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
                input_text_special, 
                return_tensors = "pt", 
                max_length = tokenizer.model_max_length, 
                truncation = True, 
                padding = False, 
            ).input_ids 
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

        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)

        return {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": ((inputs != tokenizer.pad_token_id)).to(dtype=torch.int), 
            # ((inputs != tokenizer.pad_token_id) & (inputs != labels)).to(dtype=torch.int)
        }

    # TODO Step 5: Define the Trainer
    # HINT: https://huggingface.co/docs/transformers/main_classes/trainer
    trainer = Trainer(
        # ...,
        # model=model,
        ### cjy
        model=model, # 模型  (PreTrainedModel or torch.nn.Module)
        args=training_args, # 训练参数 (TrainingArguments)
        data_collator=data_collator, # 数据整理器 (DataCollator)
        train_dataset=dataset['train'], # 训练数据集 (Dataset)
        tokenizer=tokenizer,
        ### cjy
    )
    # Step 6: Train!
    trainer.train()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model.")
    parser.add_argument('--mode', type=str, default='train', help='Mode of operation: train or eval')
    parser.add_argument('--gpu_ids', type=str, default='4,5,6,7', help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--version', type=str, default='0.5B', help='Version of the model')
    parser.add_argument('--note', type=str, default='', help='Note for the model')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    mode = args.mode

    # 创建一个字典来存储配置参数
    config_dict = {
        "model_name_or_path": "/home/xiaxinyuan/.cache/kagglehub/models/qwen-lm/qwen2.5/transformers/0.5b/1/",
        "dataset_path": "/home/xiaxinyuan/.cache/kagglehub/datasets/thedevastator/alpaca-language-instruction-training/versions/2/train.csv",
        # "dataset_path": "peft/small.csv",
        # "torch_dtype": "float32",
        "output_dir": f"/ssd/xiaxinyuan/code/CS3602_NLP_Final_Project/output/{args.version}",
        "remove_unused_columns": "False",
        "max_length": "512",
        "skip_too_long": "True",
        "learning_rate": "5e-6",
        "lr_scheduler_type": "cosine",
        "optim": "adamw_hf",
        "warmup_ratio": "0.03",
        "weight_decay": "0.003",
        "per_device_train_batch_size": "4",
        "per_device_eval_batch_size": "4",
        "save_steps": "10000",
        "save_total_limit": "4",
        "num_train_epochs": "5",
        "fp16": "True",
        "overwrite_output_dir": "True",
        "seed": "42",
        "version": args.version,
        "note": args.note,
        
    }

    # 更新sys.argv以包含配置参数
    sys.argv = [
        "notebook",
        "--model_name_or_path", config_dict["model_name_or_path"],
        "--dataset_path", config_dict["dataset_path"],
        # "--torch_dtype", config_dict["torch_dtype"],
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
        # "--fp16", config_dict["fp16"],
        "--overwrite_output_dir", config_dict["overwrite_output_dir"],
        "--seed", config_dict["seed"],
    ]
    
    with open('wandb.txt', 'r') as file:
        wandb_key = file.read()
    if mode == 'train':
        wandb.login(key=wandb_key)
        # 使用config_dict而不是sys.argv
        wandb.init(project="CS3602_NLP_Final_Project", config=config_dict, tags=["Qwen2.5-0.5B"])
    elif mode == 'debug':
        wandb.init(mode='disabled')
    else:
        raise ValueError(f"Invalid mode: {mode}")
    finetune()

if __name__ == "__main__":
    main()