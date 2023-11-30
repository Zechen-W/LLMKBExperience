# python run_llm.py \
#     --source=data/source/nq.json \
#     --usechat \
#     --type=generate \
#     --model=llama \
#     --gpu=1 \
#     --ra=none \
#     --outfile=data/source/nq-llama.json

# OPENAI_API_KEY=[your api key] \
python run_llm.py \
    --source=data/source/nq-llama.json \
    --usechat \
    --type=qa \
    --model=llama \
    --gpu=1 \
    --ra=sparse \
    --outfile=data/qa/nq-sparse-qa-llama.json