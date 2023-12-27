import pdb
import jsonlines
import random

# with open(
#     "/home/wzc2022/dgt_workspace/LLM-Knowledge-alignment-dgt/data/source/nq.json"
# ) as fin, jsonlines.open("./data/qa_pair.jsonl", "w") as fout:
#     line_id = 0
#     while True:
#         line = fin.readline()
#         if line == "":
#             break
#         line = eval(line)
#         q_id = line["id"]
#         question = line["question"]
#         ctx_set = set()
#         for ctx in line["dense_ctxs"] + line["sparse_ctxs"]:
#             if ctx not in ctx_set:
#                 ctx_set.add(ctx)
#                 fout.write(dict(line_id=line_id, q_id=q_id, question=question, ctx=ctx))
#                 line_id += 1

with jsonlines.open("./data/qa_pair.jsonl", "r") as fin, jsonlines.open(
    "./data/part_qa_pair.jsonl", "w"
) as fout:
    random.seed(0)
    line_list = []
    q_id = -1
    # pdb.set_trace()
    for line in fin.iter():
        if line["q_id"] == q_id:
            line_list.append(line)
        else:
            # pdb.set_trace()
            random.shuffle(line_list)
            for item in line_list[:3]:
                fout.write(item)
            q_id = line["q_id"]
            line_list = [line]
