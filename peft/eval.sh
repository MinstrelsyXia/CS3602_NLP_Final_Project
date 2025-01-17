# hf_path="output/peft_3b/checkpoint-30000"
hf_path='/home/xiaxinyuan/.cache/kagglehub/models/qwen-lm/qwen2.5/transformers/3b-instruct/1'
work_dir="/ssd/xiaxinyuan/code/CS3602_NLP_Final_Project/evals/safety"
opencompass \
    --datasets truthfulqa_gen  \
    --summarizer example \
    --hf-type base \
    --hf-path $hf_path \
    --tokenizer-kwargs padding_side="left" truncation="left" \
    --max-seq-len 2048 \
    --batch-size 4 \
    --hf-num-gpus 8 \
    --work-dir $work_dir \
    --debug

# truthfulqa_gen civilcomments_clp crowspairs_gen_02b6c1  cvalues_responsibility_gen_543378 jigsaw_multilingual_clp_1af0ae 
# truthfulqa_gen