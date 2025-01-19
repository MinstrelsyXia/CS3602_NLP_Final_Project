# hf_path="output/peft_3b/checkpoint-30000" # 3b-peft
hf_path="output/sft_0.5b_p/checkpoint-50000" # 0.5b-sft
# hf_path='/home/xiaxinyuan/.cache/kagglehub/models/qwen-lm/qwen2.5/transformers/0.5b/1/'
# hf_path="/home/xiaxinyuan/.cache/kagglehub/models/qwen-lm/qwen2.5/transformers/3b/1/" # 3b
# hf_path="/home/xiaxinyuan/.cache/kagglehub/models/qwen-lm/qwen2.5/transformers/3b-instruct/1" # 3b-instruct
work_dir="/ssd/xiaxinyuan/code/CS3602_NLP_Final_Project/evals/all"
opencompass \
    --datasets SuperGLUE_BoolQ_few_shot_ppl commonsenseqa_ppl hellaswag_clean_ppl race_ppl \
    --summarizer example \
    --hf-type base \
    --hf-path "${hf_path}" \
    --tokenizer-kwargs padding_side="left" truncation="left" \
    --max-seq-len 2048 \
    --batch-size 4 \
    --hf-num-gpus 1 \
    --work-dir "${work_dir}" \
    --debug

# truthfulqa_gen civilcomments_clp crowspairs_gen_02b6c1  cvalues_responsibility_gen_543378 jigsaw_multilingual_clp_1af0ae 
# truthfulqa_gen
# mmlu_ppl hellaswag_clean_ppl winogrande_ll ARC_e_ppl ARC_c_clean_ppl SuperGLUE_BoolQ_few_shot_ppl

# SuperGLUE_BoolQ_few_shot_ppl commonsenseqa_ppl hellaswag_clean_ppl storycloze_ppl piqa_ppl siqa_ppl tydiqa_gen flores_gen gsm8k_gen race_ppl teval_en_gen 

# 常识问题 commonsenseqa, natural_question
# 常识推理：hellaswag_clean_ppl
# 翻译：tydiqa

# success: SuperGLUE_BoolQ_few_shot_ppl
# fail: gsm8k_gen 
# commonsenseqa_ppl hellaswag_clean_ppl 
# teval_en_gen 