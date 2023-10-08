from datasets import Dataset, DatasetDict
corpus = '../llm_rankers/data/msmarco/collection.dl20_54.tsv'
queries = '../llm_rankers/data/msmarco/msmarco-test2020-queries_54.tsv'
qrels = '../llm_rankers/data/msmarco/2020qrels-pass.txt'

dataset_name = 'trec_dl20'

dataset_name = 'trec_dl19'
corpus = 'collection_2019.tsv'
queries = '/projects/0/gusr0546/data_rank_model/msmarco/msmarco-test2019-queries_43.tsv'
qrels = '/projects/0/gusr0546/data_rank_model/msmarco/2019qrels-pass.txt'

def load(fname):
    d = {"id_" : [], 'text': []}
    for l in open(fname):
        id_, text = l.strip().split('\t')
        d['id_'].append(id_)
        d['text'].append(text)
    return d

def load_qrel(fname):
    d = {"query-id" : [], 'corpus-id': [], 'score': []}
    for l in open(fname):
        qid, q0, did, score = l.strip().split()
        d['query-id'].append(qid)
        d['corpus-id'].append(did)
        d['score'].append(score)
    return d

queries = load(queries)
corpus = load(corpus)
dataset_hf = DatasetDict({'queries': Dataset.from_dict(queries), 'corpus': Dataset.from_dict(corpus) })
dataset_hf = DatasetDict(dataset_hf)
dataset_hf.push_to_hub(f'dmrau/{dataset_name}')

qrels = load_qrel(qrels)
qrels_hf = Dataset.from_dict(qrels)

dataset_hf = DatasetDict({'test': qrels_hf})
dataset_hf.push_to_hub(f'dmrau/{dataset_name}-qrels')



