for num in {0..15} ; do
	echo $num
	#./trec_eval qrels/qrels_trec_dl20.txt reranking_prompts/beir_bm25_runs_top100_trec_dl20_bigscience_T0_3B_prompt_$num.trec -l 2 -m recip_rank  -l 2 -M 10
	./trec_eval qrels/qrels_trec_dl20.txt reranking_prompts/beir_bm25_runs_top100_trec_dl20_bigscience_T0_3B_prompt_$num.trec -l 2 -m ndcg_cut.10 -l 2 -M 10


done
