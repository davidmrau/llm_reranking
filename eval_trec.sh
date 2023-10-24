
./trec_eval qrels/qrels_trec_dl20.txt $1 -m ndcg_cut.10 -l 2
./trec_eval qrels/qrels_trec_dl20.txt $1 -m map
