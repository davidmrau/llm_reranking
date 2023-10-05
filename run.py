from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, T5ForConditionalGeneration
from collections import defaultdict
import evaluate
from metrics import Trec

bm25_runs = "beir_bm25_runs/"
template_run_file = "run.beir.bm25-multifield.{}.txt" 


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
    examples = dataset[split_name]
    mapping = {}
    for example in tqdm(examples):
        mapping[example["_id"]] =  example["title"] + " " + example["text"]
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
    return f"""write a question based on this text.
text:
{sample['document']}
question:
{sample['query']}"""


def load_model_and_tokenizer(model_name):
    try:
        quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype='float16',
        bnb_4bit_use_dobule_quant=False
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=quant_config, use_flash_attention_2=True)
    except:
        try:
            print('- ' * 10 + ' Quantization and Flash Attention  2.0 not used! ' + '- ' * 10)
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        except:
            print('- ' * 10 + 'Using T5 model' +  '- ' * 10)
            model = T5ForConditionalGeneration.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right',trust_remote_code=True)
    return model, tokenizer


def collate_fn(batch):
    qids = [sample['qid'] for sample in batch]
    dids = [sample['did'] for sample in batch]
    #target = [sample['gen_rel_document'] for sample in batch]
    target = ['\n' + sample['query'] for sample in batch]
    instr = [format_instruction(sample)  for sample in batch]  # Add prompt to each text
    #instr_tokenized = tokenizer(instr, padding=True, truncation=True, return_tensors="pt")
    #target_tokenized = tokenizer(target, padding=True, truncation=True, return_tensors="pt").input_ids[..., 3:]
    instr_tokenized = tokenizer(instr, truncation=True, return_tensors="pt")
    target_tokenized = tokenizer(target, truncation=True, return_tensors="pt").input_ids[..., 3:]
    #remove_token_type_ids(instr_tokenized)
    return qids, dids, instr_tokenized, target_tokenized




def get_scores(model, instr_tokenized, target_tokenized):
    #remove_token_type_ids(instr_tokenized)
    logits = model(**instr_tokenized.to('cuda')).logits
    #loss_fct = CrossEntropyLoss(reduction='none', ignore_index=model.config.pad_token_id)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    target = target_tokenized.to('cuda')
    logits_target = logits[:, -(target_tokenized.shape[1]+1):-1 :].permute(0, 2, 1)
    loss = loss_fct(logits_target, target)
    return -loss.mean(1).unsqueeze(1)



def get_scores(model, instr_tokenized, target_tokenized):
    remove_token_type_ids(instr_tokenized)
    logits = model(input_ids=instr_tokenized.to('cuda').input_ids, attention_mask=instr_tokenized.to('cuda').attention_mask, labels=instr_tokenized.to('cuda').input_ids).logits
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    loss = -log_softmax.gather(2, target_tokenized.unsqueeze(2)).squeeze(2) 
    loss = torch.sum(loss, dim=1)
    return loss

dataset_name = 'trec-covid'
model_name = 'gpt2'
model_name = 'bigscience/T0_3B'
#load data
corpus = ds_to_map(load_dataset(f'BeIR/{dataset_name}', 'corpus'), 'corpus')
queries = ds_to_map(load_dataset(f'BeIR/{dataset_name}', 'queries'), 'queries')
trec_run = load_trec_run(bm25_runs + template_run_file.format(dataset_name))

#qrels = trectools_qrel(f'BeIR/{dataset_name}-qrels')

qrels_hf = load_dataset(f'BeIR/{dataset_name}-qrels')
qrels_file = dump_qrels(dataset_name, qrels_hf['test'], folder='qrels')
trec = evaluate.load("trec_eval")
ranking_file = 'reranking/{dataset_name}_{model_name}.trec'
batch_size = 1
dataset = make_hf_dataset(trec_run, corpus, queries)
# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)

model, tokenizer = load_model_and_tokenizer(model_name)

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

    
