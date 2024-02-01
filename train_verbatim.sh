accelerate launch finetuning.py \
    --output_dir="verbatim_finetune" \
    --train_file="verbatim_train.jsonl" --train_type="causal_lm_loss" \
    --model_name_or_path="microsoft/Llama2-7b-WhoIsHarryPotter" \
    --do_train --remove_unused_columns=False \
    --use_lora=True --lora_target_modules 'q_proj' 'k_proj' 'v_proj' 'o_proj' 'up_proj' 'down_proj' --lora_dimension=8 \
    --logging_steps 50 --per_device_train_batch_size=8 \
    --torch_dtype=bfloat16 --bf16=True --save_steps=100 \
    --label_names="labels" --num_train_epochs=1 \
    --weight_decay=0.01 --learning_rate=1e-5 \