accelerate launch finetuning.py \
    --output_dir="testing" \
    --train_file="verbatim_train.jsonl" --data_collator="default_data_collator" \
    --model_name_or_path="microsoft/Llama2-7b-WhoIsHarryPotter" \
    --do_train --remove_unused_columns=False \
    --use_lora=True --lora_target_modules 'down_proj' 'o_proj' --lora_dimension=8 \
    --logging_steps 50 --per_device_train_batch_size=2 \
    --torch_dtype=bfloat16 --bf16=True