### This is the resource to reproduce our primary experiment that comparing the benefit and detriment at token-level.

```
/app/anaconda3/bin/deepspeed --num_gpus=1 --master_addr="127.0.0.0" --master_port=29560 comparing_benefit_and_detriment.py \
   --file_path wikitext103 \ # Your dataset path, such as wikitext103
   --model_name_or_path $Your_LLM_path \ # Your LLm path
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \ 
```