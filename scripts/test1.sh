# OPENAI_API_KEY=`cat /home/wzc2022/dgt_workspace/LLM-Knowledge-alignment-dgt/OPENAI_API_KEY.txt`\


# # generate
# python run_llm.py \
#     --source=data/source/nq.json \
#     --usechat \
#     --type=generate \
#     --ra=none \
#     --outfile=data/source/nq-chat.json

# OPENAI_API_KEY=[your api key] \
python run_llm.py \
    --source=data/source/nq-chat.json \
    --usechat \
    --type=qa \
    --ra=none \
    --outfile=data/qa/nq-none-qa.json