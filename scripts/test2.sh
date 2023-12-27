# python run_llm.py \
#     --source=data/source/nq-chat.json \
#     --usechat \
#     --type=qa \
#     --ra=none \
#     --outfile=data/qa/nq-none-qa.json


python run_llm.py \
    --source=data/source/nq-llama.json \
    --usechat \
    --type=qa \
    --ra=sparse \
    --rank \
    --model=llama2 \
    --outfile=data/qa/nq-sparse-llama2_test.json