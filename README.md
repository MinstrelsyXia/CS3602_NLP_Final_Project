# CS3602_NLP_Final_Project
CS3602_NLP Final Project

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

