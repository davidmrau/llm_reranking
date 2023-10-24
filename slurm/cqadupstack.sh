
for data in cqadupstack-android cqadupstack-english cqadupstack-gaming cqadupstack-gis cqadupstack-mathematica cqadupstack-physics cqadupstack-programmers cqadupstack-stats cqadupstack-tex cqadupstack cqadupstack-unix cqadupstack-webmasters cqadupstack-wordpress; do

	source ../llm_rankers/venv/bin/activate
	python3 rerank_t5.py $data
done

