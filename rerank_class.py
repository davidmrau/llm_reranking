from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, T5ForConditionalGeneration
from collections import defaultdict
import evaluate
from metrics import Trec
import sys
import os

def load_trec_run(fname):
    trec_run_data = []
    for line in open(fname):
        parts = line.strip().split()
        qid = parts[0]
        did = parts[2]
        trec_run_data.append((qid, did))
    return trec_run_data

# to dict
def ds_to_map(dataset, split_name):
    print(dataset, split_name)
    examples = dataset[split_name]
    mapping = {}
    for example in tqdm(examples):
        if 'title' in example:
            content = example['title'] + " " + example['text']
        else:
            content = example['text']
        mapping[example["_id"]] =  content
    return mapping

def make_hf_dataset(trec_run, corpus, queries):
    data = defaultdict(list)
    for qid, did in trec_run:
        query, doc = queries[qid], corpus[did]
        data['query'].append(query)
        data['document'].append(doc)
        data['did'].append(did)
        data['qid'].append(qid)
    return Dataset.from_dict(data)


def trectools_qrel(qrels_name):
    qrels_hf = load_dataset(qrels_name)
    data = qrels_hf['test']
    return {'query': data['query-id'], 'docid': data['corpus-id'], 'rel': data['score']}

def dump_qrels(dataset_name, hf_dataset, folder='qrels'):
    qrels_file = f'{folder}/qrels_{dataset_name}.txt'
    with open(qrels_file, 'w') as fout:
        for el in hf_dataset:
            score = el['score']
            did = el['corpus-id']
            qid = el['query-id']
            fout.write(f'{qid}\t0\t{did}\t{score}\n')

    return qrels_file




def format_instruction(sample):
    return f"""Does the text answer the question? Question: {sample['query']} Text: {sample['document']} Answer:"""


def format_instruction(sample):
    return f"""### Instruction:
Does the text answer the question?
### Question:
{sample['query']}
### Text:
{sample['document']}
### Answer:"""
def load_model_and_tokenizer(model_name):
    try:
        quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype='float16',
        bnb_4bit_use_dobule_quant=False
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=quant_config, use_flash_attention_2=True, )
        #model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=quant_config, low_cpu_mem_usage = True)
    except:
        try:
            print('- ' * 10 + ' Quantization and Flash Attention  2.0 not used! ' + '- ' * 10)
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        except:
            print('- ' * 10 + 'Using T5 model' +  '- ' * 10)
            model = T5ForConditionalGeneration.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left',trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    model.config.pretraining_tp = 1
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.model.embed_tokens.padding_idx = len(tokenizer) - 1
    model.model.embed_tokens._fill_padding_idx_with_zero()

    return model, tokenizer


def collate_fn(batch):
    qids = [sample['qid'] for sample in batch]
    dids = [sample['did'] for sample in batch]
    target = [sample['query'] for sample in batch]
    instr = [format_instruction(sample)  for sample in batch]  # Add prompt to each text
    instr_tokenized = tokenizer(instr, padding=True, truncation=True, return_tensors="pt")
    target_tokenized = tokenizer(target, padding=True, truncation=True, return_tensors="pt")
    #remove_token_type_ids(instr_tokenized)
    #remove_token_type_ids(target_tokenized)
    return qids, dids, instr_tokenized, target_tokenized







def get_scores(model, instr_tokenized, target_tokenized):
    true_tokenid = tokenizer.encode('\n' + 'true', add_special_tokens=False)[-1]
    false_tokenid = tokenizer.encode('\n' + 'false', add_special_tokens=False)[-1]
    #remove_token_type_ids(instr_tokenized)
    with torch.cuda.amp.autocast():
        scores = model.generate(**instr_tokenized.to('cuda'), max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True).scores
    scores = torch.stack(scores)
    scores = scores[0, :, [false_tokenid, true_tokenid]].float()
    true_prob = torch.softmax(scores, 1)[:, 1]
    return true_prob


def rerank(dataset_name, model, tokenizer, bm25_runs, batch_size, ranking_file, hf_user):
    #load data
    template_run_file = "run.beir.bm25-multifield.{}.txt"
    qrels_hf = load_dataset(f'{hf_user}/{dataset_name}-qrels')
    split = 'validation' if 'dataset_name' == 'msmarco' else 'test'
    qrels_file = dump_qrels(dataset_name, qrels_hf[split], folder='qrels')
    if 'dmrau' == hf_user:
        queries = ds_to_map(load_dataset(f'{hf_user}/{dataset_name}'), 'queries')
        corpus = ds_to_map(load_dataset(f'{hf_user}/{dataset_name}'), 'corpus')
    else:
        queries = ds_to_map(load_dataset(f'{hf_user}/{dataset_name}', 'queries'), 'queries')
        corpus = ds_to_map(load_dataset(f'{hf_user}/{dataset_name}', 'corpus'), 'corpus')
    trec_run = load_trec_run(bm25_runs + '/' + template_run_file.format(dataset_name))

    qrels = trectools_qrel(f'{hf_user}/{dataset_name}-qrels')

    dataset = make_hf_dataset(trec_run, corpus, queries)
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,  num_workers=4)

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
    test = Trec('ndcg_cut_10', 'trec_eval', qrels_file, 100, ranking_file_path=ranking_file)
    eval_score = test.score(sorted_scores, q_ids)
    print('ndcg_cut_10', eval_score)


#dataset_names = ['scifact', 'scidocs', 'nfcorpus', 'fiqa', 'trec-covid', 'webis-touche2020', 'nq', 'msmarco', 'hotpotqa', 'arguana', 'quora', 'dbpedia-entity', 'fever', 'climate-fever', 'trec_dl20', 'trec_dl19']


dataset_name = sys.argv[1]

model_name = sys.argv[2]
model, tokenizer = load_model_and_tokenizer(model_name)
batch_size = 32
#bm25_runs = "beir_bm25_runs_top100"
bm25_runs = "beir_bm25_runs_top100"


print(dataset_name)
ranking_file = f'class_reranking_llama_bz_1/{bm25_runs}_{dataset_name}_{model_name.replace("/", "_")}'
if not os.path.exists(ranking_file):
    print(ranking_file)
    hf_user = 'dmrau' if 'trec_dl' in dataset_name or 'cqadupstack' in dataset_name  else 'BeIR'
    rerank(dataset_name, model, tokenizer, bm25_runs, batch_size, ranking_file, hf_user)


