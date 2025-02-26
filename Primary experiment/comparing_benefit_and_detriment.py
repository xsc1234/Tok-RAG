#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys,json
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from my_transformers import (
    SchedulerType,
    get_scheduler,
)
from my_transformers.models.llama.modeling_llama import LlamaForCausalLM

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.utils.data import Dataset
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.utils import print_rank_0, to_device, save_hf_format_dual, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from model.model import create_hf_model
import transformers
import subprocess
import copy
import regex,string
from collections import Counter
IGNORE_INDEX = -100
from dataclasses import dataclass
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `6,2,2`'
                        'will use 60%% of data for phase 1, 20%% for phase 2'
                        'and 20%% for phase 3.')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--file_path",
        type=str,
        help=
        "Path to the dataset.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

def tokenize(
        prompt,
        completion,
        tokenizer: transformers.PreTrainedTokenizer,
):
    """Preprocess the data by tokenizing."""
    source_output = tokenizer.encode(prompt)
    input_seq = prompt + ' ' + completion
    passage_list = prompt
    tokenize_output = tokenizer(input_seq, padding=False, return_tensors=None,truncation=False)
    passage_list_tokenize_output = tokenizer(passage_list, padding=False, return_tensors=None, truncation=False)
    IGNORE_INDEX = -100
    source_len = len(source_output) - 1
    tokenize_output["labels"] = copy.deepcopy(tokenize_output["input_ids"])
    tokenize_output["labels"] = [IGNORE_INDEX] * source_len + tokenize_output["labels"][source_len:]
    return passage_list_tokenize_output,tokenize_output

special_token_list = [1,32000,32001]
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_type,
                 data_list):
        super(SupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_list = data_list

        if data_type == 'train':
            self.data_list = self.data_list[:int(1.0*len(self.data_list))]
        else:
            self.data_list = self.data_list[int(0.2*len(self.data_list))+1:]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        if i % (1000) == 0 and int(os.environ.get("LOCAL_RANK")) == 0:
            sp = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out_str = sp.communicate()
            for out_element in out_str:
                for line in str(out_element).split('\\n'):
                    print(line, file=sys.stderr)

        return self.data_list[i]
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):

        # data_list = instances
        # instances_new = []
        return instances

def cut_and_get_batch(instances_new,setting_max_length,tokenizer):
    input_ids = []
    attention_mask = []
    labels = []
    max_length = 0
    data_tag = []
    prefix_len = []
    target_len = []
    rag_len = []
    for instance in instances_new:
        if len(instance["input_ids"]) > max_length:
            max_length = min(setting_max_length, len(instance["input_ids"]))
        if len(instance["input_ids"]) > setting_max_length:
            max_length = setting_max_length
            instance["input_ids"] = instance["input_ids"][:setting_max_length]
            instance["attention_mask"] = instance["attention_mask"][:setting_max_length]
            instance["labels"] = instance["labels"][:setting_max_length]
        input_ids.append(instance["input_ids"])
        attention_mask.append(instance["attention_mask"])
        labels.append(instance["labels"])
        data_tag.append(instance['data_tag'])
        prefix_len.append(instance['prefix_len'])
        target_len.append(instance['target_len'])
        rag_len.append(instance['rag_len'])

    for i in range(len(input_ids)):
        # print(data_tag, len(input_ids[i]), len(labels[i])) #label和input id不想等
        if not len(input_ids[i]) == len(labels[i]):
            print(input_ids[i])
            print(labels[i])
        remainder_pad = [tokenizer.pad_token_id] * (max_length - len(input_ids[i]))
        remainder_att = [0] * (max_length - len(input_ids[i]))
        remainder_ign = [IGNORE_INDEX] * (max_length - len(labels[i]))
        if tokenizer.padding_side == 'left':
            input_ids[i] = remainder_pad + input_ids[i]
            attention_mask[i] = remainder_att + attention_mask[i]
            labels[i] = remainder_ign + labels[i]
        elif tokenizer.padding_side == 'right':
            input_ids[i] = input_ids[i] + remainder_pad
            attention_mask[i] = attention_mask[i] + remainder_att
            labels[i] = labels[i] + remainder_ign
        else:
            raise NotImplementedError('Invalid padding-side setup! Two choices only: left and right. ')
    # input_ids, attention_mask, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "attention_mask", "labels"))
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels, dtype=torch.long)
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        data_tag=data_tag,
        prefix_len=prefix_len,
        target_len=target_len,
        rag_len=rag_len
    )
    return batch



def get_dola(bs_idx,tokenizer,candidate_premature_layers,dict_outputs,mature_layer,prefix_len,input_ids_len):
    premature_layer_dist = {l: 0 for l in candidate_premature_layers}
    premature_layers = []
    dic_token_to_js = {}
    dola = 0
    for seq_i in range(prefix_len, input_ids_len):
        # Pick the less like layer to contrast with
        # 1. Stacking all premature_layers into a new dimension
        stacked_premature_layers = torch.stack(
            [dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)

        # 2. Calculate the softmax values for mature_layer and all premature_layers
        softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :],
                                         dim=-1)  # shape: (batch_size, num_features)
        softmax_premature_layers = F.softmax(stacked_premature_layers,
                                             dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

        # 3. Calculate M, the average distribution
        M = 0.5 * (softmax_mature_layer[None, :,
                   :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)

        # 4. Calculate log-softmax for the KL divergence
        log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :],
                                                 dim=-1)  # shape: (batch_size, num_features)
        log_softmax_premature_layers = F.log_softmax(stacked_premature_layers,
                                                     dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

        # 5. Calculate the KL divergences and then the JS divergences
        kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(
            -1)  # shape: (num_premature_layers, batch_size)
        kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(
            -1)  # shape: (num_premature_layers, batch_size)
        js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

        # 6. Reduce the batchmean
        js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
        next_tokens = torch.argmax(softmax_mature_layer, dim=-1)
        output = tokenizer.decode(next_tokens)
        # print(seq_i)
        # print(output)
        dic_token_to_js[seq_i] = {output: js_divs}
        premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
        #print('max premature layer is {}'.format(premature_layer))
        dola = js_divs.max().cpu().item()
        premature_layer_dist[premature_layer] += 1

        premature_layers.append(premature_layer)

    base_logits = dict_outputs[premature_layers[0]][0, prefix_len]

    final_logits = dict_outputs[mature_layer][0, prefix_len]

    final_logits = final_logits.log_softmax(dim=-1)
    base_logits = base_logits.log_softmax(dim=-1)
    diff_logits = final_logits - base_logits

    diff_logits = diff_logits.log_softmax(dim=-1)

    return diff_logits,dola

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    norm_predict = normalize_answer(prediction)
    norm_answer = normalize_answer(ground_truth)
    return float(norm_answer in norm_predict)

def eval_ans(prediction: str, reference: str):
    norm_pred, norm_ref = normalize_answer(prediction), normalize_answer(reference)
    em = exact_match_score(prediction,reference)

    zeros = (0., 0., 0.)
    pred_tokens, ref_tokens = norm_pred.split(), norm_ref.split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return (em,) + zeros
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return em, f1, precision, recall

def true_ot_not(output,ground_truth):
    answer_list = ground_truth
    for answer in answer_list:
        em, f1, precision, recall = eval_ans(output, answer)
        if em > 0:
            return True
    return False

def right_or_not_ref(data,ref):
    if true_ot_not(ref, data['answer']):
        return True
    return False

def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step1_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
    ds_config['fp16']['enabled'] = False
    print(ds_config)
    ds_config['bf16'] = {'enabled': True}
    print(ds_config)

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    print('args.model_name_or_path is {}'.format(args.model_name_or_path))
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    tokenizer.padding_side = 'left'
    if 'llama' in args.model_name_or_path:
        tokenizer.pad_token_id = 0
    print(tokenizer.padding_side)
    print(tokenizer.pad_token_id)
    print(tokenizer.eos_token_id)
    print(tokenizer.bos_token_id)
    print('len tokenizer is')
    print(len(tokenizer))

    model = create_hf_model(LlamaForCausalLM,
                            args.model_name_or_path,
                            tokenizer, None)
    word_embedding_layer = model.model.embed_tokens

    # prepare nq data
    import joblib
    file = args.file_path
    data_list = joblib.load(file)
    train_dataset = SupervisedDataset(tokenizer=tokenizer,data_type='train',data_list=data_list)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer,data_type='dev',data_list=data_list)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    if args.local_rank == -1:
        train_sampler = SequentialSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  shuffle=False,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    print_rank_0("***** Running *****", args.global_rank)
    setting_max_length = 750
    label_list = []
    pre_list = []
    now_layer = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        acc_class = 0
        true_pos = 0
        true_pos_right_or_false = 0
        pos_count = 0
        pos_count_right_or_false = 0
        true_false = 0
        true_false_right_or_false = 0
        false_count = 0
        false_count_right_or_false = 0
        sum_tokens = 0
        acc_class_right_or_false = 0
        sum_tokens_right_or_false = 0
        conflict_sum = 0

        wudao_max_val_count = 0
        meiwudao_max_val_count = 0

        layer_count = 0
        top1_score_ir_lose_llm_win = {'ir':0,'llm':0}

        top1_score_ir_lose_llm_win_count = 1
        last_data = None
        for step, data in enumerate(train_dataloader):
            if step % (10) == 0 and int(os.environ.get("LOCAL_RANK")) == 0:
                sp = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out_str = sp.communicate()
                for out_element in out_str:
                    for line in str(out_element).split('\\n'):
                        print(line, file=sys.stderr)
            ### data start ###
            instances_new = []
            instances_no_rag = []
            instances_passage = []
            for i in range(len(data)):
                data_tag = 2
                pos_list = [text['text'] for text in data[i]['list']]
                pos_text = ''
                pos_list = pos_list[:3]
                pos_text_list = [' ']

                for pos in pos_list:
                    pos_text += ('[Reference]: ' + pos)
                    pos_text_list.append(' Reference: ' + pos)
                output_passage, output_all_token = tokenize(pos_text, '',
                                                            tokenizer)
                input_passage_ids = output_passage['input_ids']

                query_texts = data[i]['question']
                if query_texts == '':
                    query_texts = last_data['question']
                    ans_texts = last_data['answer'].split(query_texts)[-1]
                else:
                    last_data = data[i]
                    query_texts = data[i]['question']
                    ans_texts = data[i]['answer'].split(query_texts)[-1]

                query_texts = query_texts.replace(" @-@ ",'-')
                query_texts = query_texts.replace(" @,@ ", ',')
                ans_texts = ans_texts.replace(" @-@ ",'-')
                ans_texts = ans_texts.replace(" @,@ ", ',')

                query_ids = tokenize(
                    ' '+query_texts, '',
                    tokenizer)[0]['input_ids'][1:]
                ans_ids = tokenize(
                    ans_texts, '',
                    tokenizer)[0]['input_ids'][1:]
                input_ids = input_passage_ids + query_ids + ans_ids
                if len(input_ids) > setting_max_length:
                    if len(input_passage_ids) - (len(input_ids) - setting_max_length) - 10 < 0:
                        input_passage_ids = []
                    else:
                        input_passage_ids = input_passage_ids[:len(input_passage_ids) - (len(input_ids) - setting_max_length) - 10]
                input_ids = input_passage_ids + query_ids + ans_ids
                labels = [IGNORE_INDEX] * (len(input_passage_ids) + len(query_ids)) + ans_ids
                prefix_len = len(input_passage_ids) + len(query_ids)
                target_len = len(ans_ids)
                input_ids_no_rag = [1] + query_ids + ans_ids
                labels_no_rag = [IGNORE_INDEX] * (1 + len(query_ids)) + ans_ids
                prefix_len_no_rag = 1+len(query_ids)
                target_len_no_rag = len(ans_ids)
                rag_len = len(input_passage_ids)

                data_temp = {}
                data_temp['input_ids'] = input_ids
                data_temp['attention_mask'] = [1] * len(input_ids)
                data_temp['labels'] = labels
                data_temp['data_tag'] = data_tag
                data_temp['prefix_len'] = prefix_len
                data_temp['target_len'] = target_len
                data_temp['rag_len'] = rag_len
                instances_new.append(data_temp)

                data_temp = {}
                data_temp['input_ids'] = input_ids_no_rag
                data_temp['attention_mask'] = [1] * len(input_ids_no_rag)
                data_temp['labels'] = labels_no_rag
                data_temp['data_tag'] = data_tag
                data_temp['prefix_len'] = prefix_len_no_rag
                data_temp['target_len'] = target_len_no_rag
                data_temp['rag_len'] = rag_len
                instances_no_rag.append(data_temp)

                data_temp = {}
                data_temp['input_ids'] = input_passage_ids
                data_temp['attention_mask'] = [1] * len(input_passage_ids)
                data_temp['labels'] = input_passage_ids
                data_temp['data_tag'] = data_tag
                data_temp['prefix_len'] = prefix_len_no_rag
                data_temp['target_len'] = target_len_no_rag
                data_temp['rag_len'] = rag_len
                instances_passage.append(data_temp)

                if step % 5000 == 0:
                    print(tokenizer.decode(input_ids))

            batch_rag = cut_and_get_batch(instances_new,setting_max_length,tokenizer)
            batch_no_rag = cut_and_get_batch(instances_no_rag, setting_max_length, tokenizer)
            batch_passage = cut_and_get_batch(instances_passage, setting_max_length, tokenizer)

            batch_input_rag = {}
            for k, v in batch_rag.items():
                if not (k == 'scores' or k == 'data_tag' or k == 'prefix_len' or k == 'target_len' or k == 'rag_len'):
                    batch_input_rag[k] = batch_rag[k]
            batch_input_rag = to_device(batch_input_rag, device)

            batch_input_no_rag = {}
            for k, v in batch_no_rag.items():
                if not (k == 'scores' or k == 'data_tag' or k == 'prefix_len' or k == 'target_len' or k == 'rag_len'):
                    batch_input_no_rag[k] = batch_no_rag[k]
            batch_input_no_rag = to_device(batch_input_no_rag, device)

            batch_input_passage = {}
            for k, v in batch_passage.items():
                if not (k == 'scores' or k == 'data_tag' or k == 'prefix_len' or k == 'target_len' or k == 'rag_len'):
                    batch_input_passage[k] = batch_passage[k]
            batch_input_passage = to_device(batch_input_passage, device)
            batch_input_passage['labels'] = None
            candidate_premature_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                          22,23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            early_exit_layers = candidate_premature_layers

            with torch.no_grad():
                dic_output_rag, outputs = model(**batch_input_rag, early_exit_layers=early_exit_layers,output_attentions=True, use_cache=False)
            rag_logits = outputs.logits #[bs,seq_len,dim]
            rag_logits = F.softmax(rag_logits,dim=-1)
            att_logits = outputs.attentions
            with torch.no_grad():
                dic_output_no_rag, outputs = model(**batch_input_no_rag, early_exit_layers=early_exit_layers,output_attentions=True, use_cache=False)
            no_rag_logits = outputs.logits  # [bs,seq_len,dim]
            no_rag_logits = F.softmax(no_rag_logits,dim=-1)

            for bs_idx in range(rag_logits.shape[0]):
                output_passage, output_all_token = tokenize(ans_texts, '',
                                                            tokenizer)
                output_passage_token_ids = output_passage['input_ids']
                for token_idx in range(min(len(output_passage_token_ids)-1,rag_logits.shape[1]-batch_rag['prefix_len'][bs_idx],no_rag_logits.shape[1]-batch_no_rag['prefix_len'][bs_idx])):
                    values_rag, indices_rag = torch.topk(rag_logits[bs_idx,batch_rag['prefix_len'][bs_idx]-1+token_idx], 3)
                    values_no_rag, indices_no_rag = torch.topk(no_rag_logits[bs_idx, batch_no_rag['prefix_len'][bs_idx]-1+token_idx],3)

                    layer_count += 1
                    middle_att = att_logits[16][0, :, batch_rag['prefix_len'][bs_idx] - 1+token_idx,:batch_rag['rag_len'][bs_idx]]  # [32,k_len]
                    middle_att = torch.mean(middle_att, dim=0)
                    values_att, indices_att = torch.topk(middle_att, min(10,middle_att.shape[0]))
                    ids = batch_rag['input_ids'][bs_idx].to(indices_att.device)
                    tokens = ids[indices_att]
                    att_embeddins = word_embedding_layer(tokens) # torch.Size([10, 4096])
                    rag_top1_embeddings = []
                    for idx_rag_token in range(len(indices_rag)):
                        rag_top1_embeddings.append(values_rag[idx_rag_token]*word_embedding_layer(indices_rag[idx_rag_token].unsqueeze(0)))
                    rag_top1_embeddings = torch.cat(rag_top1_embeddings,dim=0)
                    rag_top1_embeddings = rag_top1_embeddings.sum(dim=0, keepdim=True)

                    cos_sim = F.cosine_similarity(rag_top1_embeddings, att_embeddins, dim=1)

                    sub_tensor = cos_sim[2:7]
                    max_val = torch.max(sub_tensor)

                    no_rag_top1_embeddings = []
                    for idx_no_rag_token in range(len(indices_no_rag)):
                        no_rag_top1_embeddings.append(values_no_rag[idx_no_rag_token]*word_embedding_layer(indices_no_rag[idx_no_rag_token].unsqueeze(0)))
                    no_rag_top1_embeddings = torch.cat(no_rag_top1_embeddings,dim=0)
                    no_rag_top1_embeddings = no_rag_top1_embeddings.sum(dim=0, keepdim=True)
                    cos_sim_no_rag = F.cosine_similarity(rag_top1_embeddings, no_rag_top1_embeddings, dim=1)

                    conflict = -1
                    if max_val > cos_sim_no_rag:
                        class_res = 0
                    else:
                        class_res = 1

                    if output_passage_token_ids[1+token_idx] == indices_rag[0] and (not indices_no_rag[0] == indices_rag[0]):
                        if class_res == 0:
                            acc_class += 1
                            true_pos += 1
                        if conflict == 1:
                            conflict_sum += 1
                        sum_tokens += 1
                        pos_count += 1
                        wudao_max_val_count += 1

                        label_list.append(0)
                        pre_list.append(class_res)

                    elif (not output_passage_token_ids[1+token_idx] == indices_rag[0]) and (indices_no_rag[0] == output_passage_token_ids[1+token_idx]):
                        top1_score_ir_lose_llm_win['llm'] += values_no_rag[0]
                        top1_score_ir_lose_llm_win['ir'] += values_rag[0]
                        top1_score_ir_lose_llm_win_count += 1
                        if class_res == 1:
                            acc_class += 1
                            true_false += 1
                        if conflict == 1:
                            conflict_sum += 1
                        sum_tokens += 1
                        false_count += 1
                        meiwudao_max_val_count += 1
                        label_list.append(1)
                        pre_list.append(class_res)

                    else:
                        if conflict == 0:
                            conflict_sum += 1

                    if output_passage_token_ids[1+token_idx] == indices_rag[0]:
                        if class_res == 0:
                            acc_class_right_or_false += 1
                            true_pos_right_or_false += 1
                        sum_tokens_right_or_false += 1
                        pos_count_right_or_false += 1

                    elif not output_passage_token_ids[1+token_idx] == indices_rag[0]:
                        if class_res == 1:
                            acc_class_right_or_false += 1
                            true_false_right_or_false += 1
                        sum_tokens_right_or_false += 1
                        false_count_right_or_false += 1

            if step % 2000 == 0 and step > 0:
                now_layer = (now_layer + 1) % 40
                acc_class = 0
                sum_tokens = 0
                true_pos = 0
                pos_count = 0
                true_false = 0
                false_count = 0
                dict_pre = {}
                dict_pre['label'] = label_list
                dict_pre['pre'] = pre_list
                data_set_name = 'wikitext103'
                model_name = 'llama-2-7b-dynamic-layer'
                joblib.dump(dict_pre, '/{}-{}-pre'.format(data_set_name, model_name))

if __name__ == "__main__":
    main()
