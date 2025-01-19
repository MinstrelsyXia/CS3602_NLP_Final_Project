# CS3602_NLP_Final_Project
CS3602_NLP Final Project

## HW-1
1. 代码组织
本仓库包含了大作业一、二的代码：
- `fork-of-notebook281e850b45.ipynb`是kaggle notebook的原始代码
- `finetune.py` 是本地部署代码，需修改`config_dict`中的`model_name_or_path`, `dataset_path`, `output_dir`
- 需要使用wandb进行可视化，请在`finetune.py`中修改`mode`为`train`，并登录wandb账号
- `utils/case_study.ipynb` 是case study代码
- `peft` 是大作业二的lora训练代码

2. 本地部署
- 对于数据集，可以基于kaggle仓库，点击`download`，下载数据集至`.cache`目录下
- 对于opencompass的测评数据集，若遇到服务器网络问题，可以尝试本地下载报错的.zip文件，后放入`.cache`目录下
- conda环境见`environment.yml`

3. checkpoint
分享内容: nlp_ckpt
链接: https://pan.sjtu.edu.cn/web/share/33128b2a2667f7c36bc6da6ba6944cf4, 提取码: qfwe

## HW-2
1. 代码组织
- `peft/`：包含LoRA微调的训练和评估脚本。
    - `peft/train.py`：用于LoRA微调的训练脚本，包含模型参数、数据参数和训练参数的配置。
    - `peft/eval.sh`：用于评估LoRA微调模型的脚本，加载数据集并进行模型评估。
- `chatbot/`：包含聊天模型的定义和实现。
    - `chatbot/chatbox_class.py`：定义了一个聊天模型类，支持外接数据库，加载预训练模型和处理对话历史。
    - `chatbot/chatbox_class_xxy.py`：定义了一个聊天模型类，支持加载预训练模型和处理对话历史。
    - `chatbot/dataset/`: 包含所有外接数据库的txt文件
        - `CJY_chat_raw.txt`：包含原始聊天记录的数据文件。
        - `CJY_chat.txt`：包含清洗后的聊天记录的数据文件。
        - `xyj.txt`：带注释版的西游记原文。
        - `hlm.txt`：带注释版的红楼梦原文。
        - `chat_clean.ipynb`：清洗聊天记录的notebook
    - `chatbot/prompt/`: 包含所有prompt文件 
        - `xxy.md`: 人格化聊天机器人的prompt
        - `prompt_fe.md`：等均为荣格八维机器人的prompt
    - `chatbot/baai_models/`: 包含所有向量器模型（受限于文件大小，未放入此目录）
    - `chatbot/model/`: 包含所有预训练模型（受限于文件大小，未放入此目录）




2. 本地部署
- 测评采用了 `truthfulqa_gen SuperGLUE_BoolQ_few_shot_ppl commonsenseqa_ppl hellaswag_clean_ppl race_ppl`这些数据集，因为只有这些在opencompass上有直接的下载链接。其中可能会出现找不到metadata的报错，请参考[此issue](https://github.com/open-compass/opencompass/issues/1760) 的解决方法重新下opencompass。测评时间较长，请耐心等待。

3. checkpoint
分享内容: nlp_ckpt
链接: https://pan.sjtu.edu.cn/web/share/33128b2a2667f7c36bc6da6ba6944cf4, 提取码: qfwe