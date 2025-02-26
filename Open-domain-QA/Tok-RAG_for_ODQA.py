#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
import json
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    SchedulerType,
    default_data_collator,
    get_scheduler,
    GenerationConfig,
)
from my_transformers.models.llama.modeling_llama import LlamaForCausalLM

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.utils.data import Dataset
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from model.model import create_hf_model
import transformers
import subprocess
from utils.data import read_nq
import copy
import regex,string
import time
from collections import Counter
IGNORE_INDEX = -100
from dataclasses import dataclass
from my_transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria
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
        '--file_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Path to dataset'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
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
    passage_list_tokenize_output = tokenizer(passage_list, padding=False, return_tensors=None,truncation=False)
    IGNORE_INDEX = -100
    source_len = len(source_output) - 1
    tokenize_output["labels"] = copy.deepcopy(tokenize_output["input_ids"])
    tokenize_output["labels"] = [IGNORE_INDEX] * source_len + tokenize_output["labels"][source_len:]
    return passage_list_tokenize_output,tokenize_output

import random
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
            self.data_list = self.data_list[:int(0.8*len(self.data_list))]
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
                    #print('local is {}'.format(int(os.environ.get("LOCAL_RANK"))), file=sys.stderr)
        rand = random.random()
        if rand > 0.8:
            prompt = 'Please tell me more about: ' + self.data_list[i][0] + ' Reference is ' + self.data_list[i][
                2] + ' Answer is: '
            answer = self.data_list[i][1] + ' </s>'
            output = tokenize(prompt, answer, self.tokenizer)
        else:
            prompt = 'Please tell me more about: ' + self.data_list[i][0] + ' Reference is ' + self.data_list[i][
                1] + ' Answer is: '
            answer = self.data_list[i][1] + ' </s>'
            # if int(os.environ.get("LOCAL_RANK")) == 0:
            #     print(prompt,answer)
            output = tokenize(prompt, answer, self.tokenizer)

        return dict(input_ids=output['input_ids'],
                    attention_mask=output['attention_mask'],
                    labels=output['labels'])
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = []
        attention_mask = []
        labels = []
        max_length = 0
        setting_max_length = 512
        for instance in instances:
            # if len(instance["input_ids"]) > setting_max_length:
            #     instance["input_ids"] = instance["input_ids"][:setting_max_length]
            #     instance["attention_mask"] = instance["attention_mask"][:setting_max_length]
            if len(instance["input_ids"]) > max_length:
                max_length = min(setting_max_length,len(instance["input_ids"]))
            if len(instance["input_ids"]) > setting_max_length:
                max_length = setting_max_length
                instance["input_ids"] = instance["input_ids"][:setting_max_length]
                instance["attention_mask"] = instance["attention_mask"][:setting_max_length]
                instance["labels"] = instance["labels"][:setting_max_length]
            input_ids.append(instance["input_ids"])
            attention_mask.append(instance["attention_mask"])
            labels.append(instance["labels"])

        for i in range(len(input_ids)):
            remainder_pad = [self.tokenizer.pad_token_id] * (max_length - len(input_ids[i]))
            remainder_att = [0] * (max_length - len(input_ids[i]))
            remainder_ign = [IGNORE_INDEX] * (max_length - len(input_ids[i]))
            if self.tokenizer.padding_side == 'left':
                input_ids[i] = remainder_pad + input_ids[i]
                attention_mask[i] = remainder_att + attention_mask[i]
                labels[i] = remainder_ign + labels[i]
            elif self.tokenizer.padding_side == 'right':
                input_ids[i] = input_ids[i] + remainder_pad
                attention_mask[i] = attention_mask[i] + remainder_att
                labels[i] = labels[i] + remainder_ign
            else:
                raise NotImplementedError('Invalid padding-side setup! Two choices only: left and right. ')
        # input_ids, attention_mask, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "attention_mask", "labels"))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels, dtype=torch.long)
        return dict(
            input_ids=input_ids,
            #labels=labels,
            labels=labels,
            attention_mask=attention_mask,
        )

import joblib

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


def right_or_not(generated,ground_truth):
    if '1. ' in generated and '2.' in generated:
        generated = generated.split('2.')[0]
    if true_ot_not(generated, ground_truth):
        return True
    return False


def right_or_not_ref(data,ref):
    if true_ot_not(ref, data['answer']):
        return True
    return False


def set_stop_words(stop_words,tokenizer):
    stop_words = stop_words
    stopping_criteria = StoppingCriteriaList()
    list_stop_word_ids = []
    for stop_word in stop_words:
        stop_word_ids = tokenizer.encode('\n' + stop_word)[3:]
        list_stop_word_ids.append(stop_word_ids)
        #print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
    stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

def has_num(text):
    number_words = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    for num in number_words:
        if num in text:
            return True
    return False

filter_words = [' ',')','-',']:', ']', '}:', ':', '▁','▁[', '▁','▁the','the','a','an',
                'The','.','[','\'','Query','',' ','▁"','.', '<s>', "'",' ',')','-',']:', ']', '}:', ':','','▁','▁[', '▁'
                ,'for','in','on','at']
number_words = ['1','2','3','4','5','6','7','8','9','0']
yiwen_words = ['what','how','who','where','when','What','How','Who','Where','When']
skip_words = ['1','2','3','4','5','6','7','8','9','0','.', '<s>', "'",' ',')','-',']:', ']', '}:', ':','','▁','▁[', '▁']

def get_predict(rag_token,indices_att,rag_ids_tensor,tokenizer,topk):
    token_window = 2
    count = 0
    for idx in indices_att:
        if tokenizer.convert_ids_to_tokens([rag_ids_tensor[idx]])[0] in skip_words:
            continue
        if count == topk:
            break
        left = max(0,idx-token_window)
        right = min(rag_ids_tensor.shape[0]-1,idx+token_window)
        idice = range(left,right+1)
        tokens_in_window = [tok.lower() for tok in tokenizer.convert_ids_to_tokens(rag_ids_tensor[idice])]
        print(rag_token,tokens_in_window)
        count += 1
        if rag_token in tokens_in_window:
            return True
        if rag_token.split('▁')[-1] in tokens_in_window:
            return True
    return False

def get_predict_emb_contrastive(rag_token,indices_att,rag_ids_tensor,tokenizer,topk,values_rag,indices_rag,word_embedding_layer,no_rag_temp):
    token_window = 2
    count = 0
    rag_top1_embeddings = []
    for idx_rag_token in range(len(indices_rag)):
        rag_top1_embeddings.append(
            values_rag[idx_rag_token] * word_embedding_layer(indices_rag[idx_rag_token].unsqueeze(0)))
    rag_top1_embeddings = torch.cat(rag_top1_embeddings, dim=0)
    rag_top1_embeddings = rag_top1_embeddings.sum(dim=0, keepdim=True)

    if no_rag_temp is None:
        no_rag_score = 0
    else:
        values_no_rag, indices_no_rag = torch.topk(no_rag_temp['logits'][0], 3)
        no_rag_top1_embeddings = []
        for idx_no_rag_token in range(len(indices_no_rag)):
            no_rag_top1_embeddings.append(
                values_no_rag[idx_no_rag_token] * word_embedding_layer(indices_no_rag[idx_no_rag_token].unsqueeze(0)))
        no_rag_top1_embeddings = torch.cat(no_rag_top1_embeddings, dim=0)
        no_rag_top1_embeddings = no_rag_top1_embeddings.sum(dim=0, keepdim=True)

        cos_sim = F.cosine_similarity(rag_top1_embeddings, no_rag_top1_embeddings, dim=1)
        no_rag_score = cos_sim

    max_rag_score = 0
    for idx in indices_att:
        if tokenizer.convert_ids_to_tokens([rag_ids_tensor[idx]])[0] in skip_words:
            continue
        if count == topk:
            break
        left = max(0,idx-token_window)
        right = min(rag_ids_tensor.shape[0]-1,idx+token_window-1)
        idice = range(left,right+1)
        tokens_in_window = [tok.lower() for tok in tokenizer.convert_ids_to_tokens(rag_ids_tensor[idice])]
        att_embeddins = word_embedding_layer(rag_ids_tensor[idice])

        for att in att_embeddins:
            cos_sim = F.cosine_similarity(rag_top1_embeddings, att.unsqueeze(0), dim=1)
            if cos_sim > max_rag_score:
                max_rag_score = cos_sim
        count += 1

    if max_rag_score > no_rag_score:
        return True
    else:
        return False


def get_max_att(rag_att,rag_ids_list):
    max_sum = -10000
    max_idx = 0
    list_att = []
    for att_idx in range(len(rag_att)):
        middle_att = rag_att[att_idx][0, :, 0, :len(rag_ids_list)]  # [32,k_len]
        middle_att = torch.mean(middle_att, dim=0)
        middle_att = torch.sum(middle_att, dim=0)
        list_att.append((att_idx,middle_att.item()))
        if middle_att > max_sum:
            max_sum = middle_att
            max_idx = att_idx
    return max_idx

def get_dynamic_layer(no_rag_dola,rag_dola,candi_layers):
    for i in range(len(candi_layers)-1):
        if rag_dola['{}-{}'.format(candi_layers[i],candi_layers[i+1])] - \
            no_rag_dola['{}-{}'.format(candi_layers[i],candi_layers[i+1])] > 5e-7 and candi_layers[i] > 10:
            return candi_layers[i]-1
    return 20

dola_list = []

def lm_score(select_layer, label_list,pre_list,pre_certainty_list,rag_text,instruction_text,question,answer,tokenizer, model, device, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0,
             temperature=0.8,**kwargs):
    with torch.no_grad():
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=1,
            **kwargs,
        )

        rag_ids = tokenize(
            rag_text, '',
            tokenizer)[0]['input_ids']

        instruction_ids = tokenize(
            instruction_text, '',
            tokenizer)[0]['input_ids']


        input_ids = torch.tensor([rag_ids + instruction_ids[1:]], dtype=torch.long).to(device)
        rag_ids_list = rag_ids
        rag_ids = torch.tensor([rag_ids], dtype=torch.long).to(device)

        # 7b
        candidate_premature_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                       23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        outputs_rag = model.generate(
            input_ids=input_ids,
            tokenizer=tokenizer,
            generation_config=generation_config,
            output_attentions=False,
            dola_decoding=True,
            candidate_premature_layers=candidate_premature_layers,
            mature_layer=None,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=15,
        )
        is_num = None
        rag_first_temp = None
        s = outputs_rag.sequences[0]
        output = tokenizer.decode(s)
        rag_answer = output.split(instruction_text)[-1]

        input_ids = torch.tensor([instruction_ids], dtype=torch.long).to(device)
        outputs = model.generate(
            input_ids=input_ids,
            tokenizer=tokenizer,
            generation_config=generation_config,
            output_attentions=False,
            dola_decoding=True,
            candidate_premature_layers=candidate_premature_layers,
            mature_layer=None,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=15,
        )
        s = outputs.sequences[0]
        output = tokenizer.decode(s)
        no_rag_answer = output.split(instruction_text)[-1]
        is_open = False
        class_res = 0 
        class_wudao = 1 
        class_wudao_cer = 1
        acc_res = 0
        acc_wudao = 0
        wudao = 0

        true_pos_res = 0
        pos_res_count = 0
        true_neg_res = 0
        neg_res_count = 0

        true_pos_wudao = 0
        pos_wudao_count = 0
        true_neg_wudao = 0
        neg_wudao_count = 0
        is_count = 0
        beak_falg = False
        word_embedding_layer = model.model.embed_tokens
        no_rag_temp = None
        for temp in outputs.list_token_and_logits:
            next_tokens = torch.argmax(temp['logits'].to(torch.float32), dim=-1)
            output = tokenizer.decode(next_tokens)
            if output == 'is':
                is_open = True
            if is_open and not output == 'is' and not output in filter_words:
                values_no_rag, indices_no_rag = torch.topk(temp['logits'][0], 10)
                no_rag_temp = temp
                break

        is_open = False
        for temp in outputs_rag.list_token_and_logits:
            next_tokens = torch.argmax(temp['logits'].to(torch.float32), dim=-1)
            output = tokenizer.decode(next_tokens)
            is_count += 1
            if is_count > 3 and not is_open: 
                beak_falg = True
                break
            if output == 'is':
                is_open = True
            if is_open and not output == 'is' and not output in filter_words:
                rag_first_temp = temp
                if rag_first_temp is not None:
                    rag_att = rag_first_temp['att']  # [1,32,1,num_tokens]
                    print(question, answer)
                    layer = select_layer
                    topk = 3
                    middle_att = rag_att[layer][0, :, 0, :len(rag_ids_list)]  # [32,k_len]
                    middle_att = torch.mean(middle_att, dim=0)
                    values_att, indices_att = torch.topk(middle_att, middle_att.shape[0])
                    rag_ids_tensor = torch.tensor(rag_ids_list, dtype=torch.long).to(device)
                    tokens = rag_ids_tensor[indices_att]
                    att_tokens = [tok.lower() for tok in tokenizer.convert_ids_to_tokens(tokens)]
                    values_rag, indices_rag = torch.topk(rag_first_temp['logits'][0], 2)
                    rag_tokens = [tok.lower() for tok in tokenizer.convert_ids_to_tokens(indices_rag)]


                    if get_predict_emb_contrastive(rag_tokens[0],indices_att,rag_ids_tensor,tokenizer,topk,values_rag=values_rag,
                                    indices_rag=indices_rag,word_embedding_layer=word_embedding_layer,no_rag_temp=no_rag_temp):
                        class_res = 1
                        class_wudao = 1
                    else:
                        class_res = 0
                        class_wudao = 0
                    if rag_first_temp is not None and no_rag_temp is not None:
                        if values_rag[0] > values_no_rag[0]:
                            class_wudao_cer = 1
                        else:
                            class_wudao_cer = 0
                        break

        if beak_falg:  
            if has_num(rag_answer) and not has_num(no_rag_answer):
                class_res = 0
                class_wudao = 0
                class_wudao_cer = 0

        if right_or_not(rag_answer, answer):
            if class_res == 1:
                acc_res = 1
                true_pos_res = 1
            pos_res_count = 1
        elif not right_or_not(rag_answer, answer):
            if class_res == 0:
                acc_res = 1
                true_neg_res = 1
            neg_res_count = 1

        if right_or_not(rag_answer, answer) and not right_or_not(no_rag_answer, answer):
            label_list.append(1)
            pre_list.append(class_wudao)
            pre_certainty_list.append(class_wudao_cer)
            if class_wudao == 1:
                acc_wudao = 1
                true_pos_wudao = 1
            wudao = 1
            pos_wudao_count = 1
        elif (not right_or_not(rag_answer, answer)) and right_or_not(no_rag_answer, answer):
            label_list.append(0)
            pre_list.append(class_wudao)
            pre_certainty_list.append(class_wudao_cer)
            if class_wudao == 0:
                acc_wudao = 1
                true_neg_wudao = 1
            wudao = 1
            neg_wudao_count = 1

    return (rag_answer,no_rag_answer,class_res),acc_res,acc_wudao,wudao,\
           true_pos_res,pos_res_count,true_neg_res,neg_res_count, \
        true_pos_wudao, pos_wudao_count, true_neg_wudao, neg_wudao_count

def evaluate(
    tokenizer,
    model,
    device,
    args,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=256,
    **kwargs,
):
    result = []
    from tqdm import tqdm
    import random
    file = args.file_path
    data_set_name = args.file_path
    data_list = joblib.load(file)
    select_layer = 0
    label_list = []
    pre_list = []
    pre_certainty_list = []
    for i in range(32):
        model_name = 'llama-2-7b'
        step = 0
        acc_res_count = 0
        acc_wudao_count = 0
        wudao_count = 0

        true_pos_res_all = 0
        pos_res_count_all = 0
        true_neg_res_all = 0
        neg_res_count_all = 0

        true_pos_wudao_all = 0
        pos_wudao_count_all = 0
        true_neg_wudao_all = 0
        neg_wudao_count_all = 0

        for data in tqdm(data_list):
                output_list = []
                if 'eli5' in file or 'wow' in file or 'asqa' in file:
                    pos_list = [text for text in data['list']]
                else:
                    pos_list = [text['text'] for text in data['list']]
                pos_text = ''
                pos_list = pos_list[:5]
                pos_text_list = [' ']
                try:
                    for pos in pos_list:
                        pos_text += ('[Reference]: ' + pos)
                        pos_text_list.append(' Reference: ' + pos)
                except:
                    continue
                ICL_prefix = ' Who got the first nobel prize in physics. The answer is Wilhelm Conrad Röntgen. ' + \
                      'Who designed the garden city of new earswick. The answer is Raymond Unwin. ' + \
                      'When was the public service commission original version of the upsc set up. The answer is October 1, 1926. ' + \
                      'When does batman gotham by gaslight come out. The answer is January 12, 2018. ' + \
                      'What do they call snowboarders in johnny tsunami. The answer is Urchins. ' + \
                      'What is the name of the college in the classic movie animal house. The answer is Faber College. ' + \
                      'Where did the french national anthem come from. The answer is Strasbourg. ' + \
                      'Where did hurricane edith make landfall in 1971. The answer is Cape Gracias a Dios. ' + \
                      'Which apostle had a thorn in his side. The answer is Paul. ' + \
                     'who is charles off of pretty little liars. The answer is Drake. ' + \
                     'where is the new england patriots stadium located. The answer is Foxborough , Massachusetts. ' + \
                     'who plays caroline on the bold and beautiful. The answer is Linsey Godfrey. ' + \
                     'where are the fruits of the spirit found in the bible. The answer is Epistle to the Galatians. ' + \
                     'who is the coach of arizona state men\'s basketball. The answer is Bobby Hurley. ' + \
                     'where is the mesophyll located in a plant. The answer is In leaves. ' + \
                     'what is the name of the college in the classic movie animal house. The answer is Faber College. ' + \
                     'who sang the song i love to love. The answer is Tina Charles. '


                ICL = ICL_prefix + """ {}. The answer """.format(data['question'])
                S = pos_text + ICL

                print(data['question'])
                output,acc_res,acc_wudao,wudao, true_pos_res,pos_res_count,true_neg_res,neg_res_count, \
            true_pos_wudao, pos_wudao_count, true_neg_wudao, neg_wudao_count = lm_score(select_layer=select_layer,
               label_list=label_list,pre_list=pre_list,pre_certainty_list=pre_certainty_list,rag_text=pos_text,instruction_text=ICL, question=data['question'],answer=data['answer'],model=model,
                                                   device=device, tokenizer=tokenizer, mode='dola')
                acc_res_count += acc_res
                acc_wudao_count += acc_wudao
                wudao_count += wudao

                true_pos_res_all += true_pos_res
                pos_res_count_all += pos_res_count
                true_neg_res_all += true_neg_res
                neg_res_count_all += neg_res_count

                true_pos_wudao_all += true_pos_wudao
                pos_wudao_count_all += pos_wudao_count
                true_neg_wudao_all += true_neg_wudao
                neg_wudao_count_all += neg_wudao_count

                print(data['answer'])

                output_list.append(output)
                dic_temp = {}
                dic_temp['question'] = data['question']
                dic_temp['answer'] = data['answer']
                dic_temp['list'] = pos_list

                dic_temp['output'] = output
                dic_temp['model'] = model_name
                result.append(dic_temp)
                if step > 0 and step % 100 == 0:
                    dict_pre = {}
                    dict_pre['label'] = label_list
                    dict_pre['pre'] = pre_list
                    dict_pre['pre_cer'] = pre_certainty_list
                    joblib.dump(dict_pre, '/{}-{}-pre'.format(data_set_name, model_name))
                step += 1
        print('now layer is {}'.format(select_layer))
        print('acc res:')
        print('{} / {} = {}'.format(acc_res_count, step, acc_res_count / step))
        print('true pos rate res:')
        print('{} / {} = {}'.format(true_pos_res_all, pos_res_count_all, true_pos_res_all / (pos_res_count_all+1)))
        print('true neg rate res:')
        print('{} / {} = {}'.format(true_neg_res_all, neg_res_count_all, true_neg_res_all / (neg_res_count_all+1)))
        print('acc wudao:')
        print('{} / {} = {}'.format(acc_wudao_count, wudao_count, acc_wudao_count / (wudao_count+1)))
        print('true pos rate wudao:')
        print('{} / {} = {}'.format(true_pos_wudao_all, pos_wudao_count_all, true_pos_wudao_all / (pos_wudao_count_all + 1)))
        print('true neg rate wudao:')
        print('{} / {} = {}'.format(true_neg_wudao_all, neg_wudao_count_all, true_neg_wudao_all / (neg_wudao_count_all + 1)))
        select_layer += 1
        label_list = []
        pre_list = []
        pre_certainty_list = []



def main():
    args = parse_args()

    device = torch.device("cuda:0")

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    print('args.model_name_or_path is {}'.format(args.model_name_or_path))
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    tokenizer.padding_side = 'left'
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
    model = model.half()
    model.to(device)
    model.eval()
    evaluate(tokenizer,model,device,args=args)

if __name__ == "__main__":
    main()
