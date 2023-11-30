# python run_llm.py \
#     --source=data/source/nq.json \
#     --usechat \
#     --type=generate \
#     --model=chatglm \
#     --gpu=0 \
#     --ra=none \
#     --outfile=data/source/nq-chatglm.json

# OPENAI_API_KEY=[your api key] \
python run_llm.py \
    --source=data/source/nq-chatglm.json \
    --usechat \
    --type=qa \
    --model=chatglm \
    --gpu=0 \
    --ra=sparse \
    --outfile=data/qa/nq-sparse-qa-chatglm.json