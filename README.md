```
pip install -r requirements.txt
```
## Run retrieval model (ColBert)
Download the corpus for retrieval from DPR
```
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```

Indext your data
```
python ./retrieval/ColBERT/index.py
```

Run the service for retrieval
```
python ./retrieval/ColBERT/server_retrieval.py
```

Retrieval for your dataset
```
python ./retrieval/retrieval_top_k_data.py
```



## Primary experiment

```
/app/anaconda3/bin/deepspeed --num_gpus=1 --master_addr="127.0.0.0" --master_port=29560 comparing_benefit_and_detriment.py \
   --file_path wikitext103 \ # Your dataset path, such as wikitext103
   --model_name_or_path $Your_LLM_path \ # Your LLm path
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \ 
```

## Open-domain QA

```
/app/anaconda3/bin/deepspeed --num_gpus=1 --master_addr="127.0.0.0" --master_port=29560 Tok-RAG_for_ODQA.py \
   --file_path NQ \ # Your dataset path, such as NQ
   --model_name_or_path $Your_LLM_path \ # Your LLm path
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \ 
```