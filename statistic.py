import jsonlines
import matplotlib.pyplot as plt

cnt = {"else": 0}
for i in range(1, 6):
    path = f"/home/wzc2022/dgt_workspace/LLM-Knowledge-alignment-dgt/data/roberta_data/part_qa_pair_scores_{i}.jsonl"
    with jsonlines.open(path) as fin:
        for line in fin.iter():
            score = line["score"]
            if score.isnumeric():
                cnt[score] = cnt.get(score, 0) + 1
            else:
                cnt["else"] = cnt["else"] + 1

keys = ["0", "1", "2", "3", "4", "5", "6", "else"]

bar = plt.bar(keys, [cnt[key] for key in keys], fc="g")
plt.bar_label(bar, label_type="edge")
plt.show()
plt.savefig("statistics.png")
