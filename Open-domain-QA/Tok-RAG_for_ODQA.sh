##!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
/app/anaconda3/bin/deepspeed --num_gpus=1 --master_addr="127.0.0.0" --master_port=29560 Tok-RAG_for_ODQA.py \
   --file_path NQ \ # Your dataset path, such as NQ
   --model_name_or_path $Your_LLM_path \ # Your LLm path
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \ 