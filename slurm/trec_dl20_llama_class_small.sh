source ../llm_rankers/venv/bin/activate

python3 rerank_class.py trec_dl19 meta-llama/Llama-2-7b-hf
python3 rerank_class.py trec_dl19 meta-llama/Llama-2-7b-chat-hf
python3 rerank_class.py trec_dl19 meta-llama/Llama-2-13b-hf
python3 rerank_class.py trec_dl19 meta-llama/Llama-2-13b-chat-hf

#python3 rerank_class.py trec_dl20 meta-llama/Llama-2-7b-hf
#python3 rerank_class.py trec_dl20 meta-llama/Llama-2-7b-chat-hf
#python3 rerank_class.py trec_dl20 meta-llama/Llama-2-13b-hf
#python3 rerank_class.py trec_dl20 meta-llama/Llama-2-13b-chat-hf
