from metrics import Trec
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, T5ForConditionalGeneration
from datasets import load_dataset, Dataset
from random import randrange
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from collections import defaultdict
import sys
import os
# Set batch size and other relevant parameters
batch_size = 1
checkpoint_dir = sys.argv[1]
if len(sys.argv) > 2:
    add_to_dir = sys.argv[2]
else:
    add_to_dir = ''
#checkpoint_dir = 'meta-llama/Llama-2-7b-chat-hf'
dataset_name = 'data/msmarco/dl2020_54.bm25.passage.ref.hf'
#dataset_name = 'data/msmarco/dl2020_single.bm25.passage.hf'

output_dir = checkpoint_dir +'_' +  dataset_name.split('/')[-1] + '_' + add_to_dir
ranking_file = output_dir  + '/run.ppl'
print('ranking_file', ranking_file)
os.makedirs(output_dir, exist_ok=True)
use_flash_attention = False
load_unmerged = False

qrels_file = 'data/msmarco/2020qrels-pass.txt'






#def format_instruction(sample):
#    return f"""### Instruction:
#Rewrite the passage.
#### Passage:
#{sample['document']}
#### Passage rewritten:
#{sample['gen_rel_document']}"""


def format_instruction(sample):
    return f"""[INST]<<SYS>>
<</SYS>> [/INST]"""



def format_instruction(sample):
    return f"[INST] write a question that this passsage could answer.\npassage:\n{sample['document']}[/INST]\nquestion:\n{sample['query']}"

def format_instruction(sample):
    return f"""### Instruction:
Write a question that this passsage could answer.
### Passage:
{sample['document']}
### Question:
{sample['query']}"""
def format_instruction(sample):
    return f"""write a question based on this text.
text:
{sample['document']}"""
def format_instruction(sample):
    return f"""Passage: {sample['document']}. Please write a question based on this passage."""
if use_flash_attention:
    # unpatch flash attention
    from llama_patch import unplace_flash_attn_with_attn
    unplace_flash_attn_with_attn()

tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir,trust_remote_code=True)
#tokenizer.add_special_tokens({"pad_token":"<pad>"})
#load unmerged
if load_unmerged:
    # load base LLM model and tokenizer
    unmerged_model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        #load_in_4bit=True,
        device_map='auto'        
    )
    model = unmerged_model.merge_and_unload()
    # Save the merged model
    #model.save_pretrained(checkpoint_dir + '_merged', safe_serialization=True)
    #tokenizer.save_pretrained(checkpoint_dir+'_merged')
else:
    quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_use_dobule_quant=False
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, device_map='auto', quantization_config=quant_config)
    except:
        #model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir, device_map='auto', quantization_config=quant_config)
        model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir, device_map='auto', torch_dtype=torch.bfloat16)
try:
    model = model.to_bettertransformer()
except:
    print(f'better transformer not implmented for model: {checkpoint_dir}')
#model.config.pretraining_tp = 1
#model.config.pad_token_id = tokenizer.pad_token_id
#model.resize_token_embeddings(len(tokenizer))
#model.model.embed_tokens.padding_idx = len(tokenizer) - 1
#model.model.embed_tokens._fill_padding_idx_with_zero()

model.config.use_cache = False


dataset = Dataset.load_from_disk(dataset_name)
sample = dataset[randrange(len(dataset))]
prompt = format_instruction(sample)

def remove_token_type_ids(inp):
    if 'token_type_ids' in inp:
        del inp['token_type_ids']


def collate_fn(batch):
    qids = [sample['qid'] for sample in batch]
    dids = [sample['did'] for sample in batch]
    #target = [sample['gen_rel_document'] for sample in batch]
    target = ['Question: ' + sample['query'] for sample in batch]
    instr = [format_instruction(sample)  for sample in batch]  # Add prompt to each text
    #instr_tokenized = tokenizer(instr, padding=True, truncation=True, return_tensors="pt")
    #target_tokenized = tokenizer(target, padding=True, truncation=True, return_tensors="pt").input_ids[..., 3:]
    instr_tokenized = tokenizer(instr, padding=True, truncation=True, return_tensors="pt")
    target_tokenized = tokenizer(target, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False).input_ids
    remove_token_type_ids(instr_tokenized)
    return qids, dids, instr_tokenized, target_tokenized



# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)


def get_scores(model, instr_tokenized, target_tokenized):
    logits = model(input_ids=instr_tokenized.to('cuda').input_ids, attention_mask=instr_tokenized.to('cuda').attention_mask, labels=target_tokenized.to('cuda')).logits
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    loss = log_softmax.gather(2, target_tokenized.unsqueeze(2)).squeeze(2) 
    #print(loss)
    #print(loss.shape)
    loss = torch.sum(loss, dim=1)
    return loss



res_test = defaultdict(dict)
with torch.inference_mode():
    for batch_inp in tqdm(dataloader): 
        qids, dids, instr, target = batch_inp
        instr_tokenized= instr.to('cuda')
        target_tokenized = target.to('cuda')
        scores = get_scores(model, instr_tokenized, target_tokenized)
        batch_num_examples = scores.shape[0]
        # for each example in batch
        for i in range(batch_num_examples):
            res_test[qids[i]][dids[i]] = scores[i].item()

    sorted_scores = []
    q_ids = []
    # for each query sort after scores
    for qid, docs in res_test.items():
        sorted_scores_q = [(doc_id, docs[doc_id]) for doc_id in sorted(docs, key=docs.get
, reverse=True)]
        q_ids.append(qid)
        sorted_scores.append(sorted_scores_q)

test = Trec('ndcg_cut_10', 'trec_eval', qrels_file, 1000, ranking_file_path=ranking_file)
eval_score = test.score(sorted_scores, q_ids)
print('ndcg_cut_10', eval_score)
