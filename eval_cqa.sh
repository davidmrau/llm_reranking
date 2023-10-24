
FOLDER=$1

deci(){
	printf "%.2f\n" $(echo "scale=0; ${1}*100" | bc)
}

for data in cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress; do
	if (echo "$data"  | fgrep -q "scores"); then
		T=2
	else
		#echo ndcg
		FILE="reranking_qld/beir_qld_runs_top100_${data}_bigscience_T0_3B.trec"
		#FILE=$1"/run.beir.qld-multifield.${data}.txt"
		#ndcg=`./trec_eval qrels/qrels_$data.txt $FILE -m ndcg_cut.10 | awk '{print $3}'`
		recall=`./trec_eval qrels/qrels_$data.txt  $FILE -m recall.30 | awk '{print $3}'`
		#deci $ndcg
		deci $recall
	fi
done


