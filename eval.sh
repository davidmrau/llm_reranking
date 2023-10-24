
FOLDER=beir_bm25_runs_top1000
FOLDER=beir_qld_runs_top100
FOLDER=reranking_qld

deci(){
	printf "%.2f\n" $(echo "scale=0; ${1}*100" | bc)
}

for data in 'scifact' 'scidocs' 'nfcorpus' 'fiqa' 'trec-covid' 'webis-touche2020' 'nq' 'msmarco' 'hotpotqa' 'arguana' 'quora' 'dbpedia-entity' 'fever' 'climate-fever'; do
	echo $data
	echo ndcg
	echo recall
	#FILE=$FOLDER/run.beir.qld-multifield.$data.txt 
	FILE=$FOLDER/"beir_qld_runs_top100_${data}_bigscience_T0_3B.trec"
	ndcg=`./trec_eval qrels/qrels_$data.txt $FILE -m ndcg_cut.10 | awk '{print $3}'`
	#recall=`./trec_eval qrels/qrels_$data.txt $FILE -m recall.30 | awk '{print $3}'`
	deci $ndcg
	#deci $recall
done
