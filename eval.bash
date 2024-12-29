hf_path="output/0.5B/checkpoint-50000"
work_dir="/ssd/xiaxinyuan/code/CS3602_NLP_Final_Project/evals/plm"
opencompass \
    --datasets mmlu_ppl hellaswag_clean_ppl winogrande_ll ARC_e_ppl ARC_c_clean_ppl SuperGLUE_BoolQ_few_shot_ppl \
    --summarizer example \
    --hf-type base \
    --hf-path $hf_path \
    --tokenizer-kwargs padding_side="left" truncation="left" \
    --max-seq-len 2048 \
    --batch-size 4 \
    --hf-num-gpus 8 \
    --work-dir $work_dir \
    --debug