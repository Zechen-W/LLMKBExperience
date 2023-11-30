import json


def calculate_average_score(file_path):
    total_em = 0
    total_f1 = 0
    total_rows = 0

    with open(file_path) as f:
        for line in f.readlines():
            item = json.loads(line)
            total_em += item["EM"]
            total_f1 += item["F1"]
            total_rows += 1

    average_em = total_em / total_rows
    average_f1 = total_f1 / total_rows

    print("Average EM score:", average_em)
    print("Average F1 score:", average_f1)


# 用法示例
calculate_average_score(
    "/home/zjx2022/dgt_workspace/LLM-Knowledge-alignment-dgt/data/qa/nq-sparse-qa.json"
)
