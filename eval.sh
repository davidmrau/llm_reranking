
FOLDER=beir_bm25_runs_top100

deci(){
	printf "%.2f\n" $(echo "scale=0; ${1}*100" | bc)
}

for data in 'scifact' 'scidocs' 'nfcorpus' 'fiqa' 'trec-covid' 'webis-touche2020' 'nq' 'msmarco' 'hotpotqa' 'arguana' 'quora' 'dbpedia-entity' 'fever' 'climate-fever'; do
	echo $data
	ndcg=`./trec_eval qrels/qrels_$data.txt $FOLDER/run.beir.bm25-multifield.$data.txt -m ndcg_cut.10 | awk '{print $3}'`
	recall=`./trec_eval qrels/qrels_$data.txt $FOLDER/run.beir.bm25-multifield.$data.txt -m recall.30 | awk '{print $3}'`
	deci $ndcg
	deci $recall
done
