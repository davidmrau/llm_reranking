from datasets import Dataset, DatasetDict
import json
import csv
datasets = ['android',  'gaming',  'mathematica', 'programmers',  'tex',   'webmasters',
'english'  ,'gis'    , 'physics' ,     'stats',        'unix',  'wordpress']
root = '/scratch-shared/cqadupstack/'


def load(fname):
    d = {"_id" : [], 'text': [], 'title': []}
    for l in open(fname):
        line = json.loads(l)
        d['_id'].append(line['_id'])
        d['text'].append(line['text'])
        if 'line' in line:
            d['title'].append(line['title'])
    
        else:
            d['title'].append('')
    return d

def load_qrel(fname):
    d = {"query-id" : [], 'corpus-id': [], 'score': []}
    reader = csv.reader(open(fname, encoding="utf-8"), 
                        delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    
    for id, row in enumerate(reader):
        qid, did, score = row[0], row[1], int(row[2])
        d['query-id'].append(qid)
        d['corpus-id'].append(did)
        d['score'].append(score)
    return d
for dataset_name in datasets:
    corpus = f'{root}/{dataset_name}/corpus.jsonl'
    queries  = f'{root}/{dataset_name}/queries.jsonl'
    qrels = f'{root}/{dataset_name}/qrels/test.tsv'

    queries = load(queries)
    corpus = load(corpus)
    dataset_hf = DatasetDict({'queries': Dataset.from_dict(queries), 'corpus': Dataset.from_dict(corpus) })
    dataset_hf = DatasetDict(dataset_hf)
    dataset_hf.push_to_hub(f'dmrau/cqadupstack-{dataset_name}')

    qrels = load_qrel(qrels)
    qrels_hf = Dataset.from_dict(qrels)

    dataset_hf = DatasetDict({'test': qrels_hf})
    dataset_hf.push_to_hub(f'dmrau/cqadupstack-{dataset_name}-qrels')



