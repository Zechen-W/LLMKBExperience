from collections.abc import Callable, Iterable, Mapping
from typing import Any
import jsonlines
from connect_openai import call_api
from tqdm import tqdm
import threading


def prompt(q, ctx):
    return f"""As a professional data annotator, I will give you a question and a context below. Please judge whether to answer the given question, in addition to the knowledge you already have, you need the given context as a supplement to answer the question. You scored on a scale of 0 to 5, where 0 means you can answer the question using only your knowledge and no context at all, and 5 means you can't answer the question using your current knowledge without context.\n

question: {q}
context: {ctx}\n

Note that your answer should contain only a number and nothing else.
"""


def ask_part(lines, n_part):
    with jsonlines.open(
        f"./data/part_qa_pair_scores_{n_part}.jsonl", "w"
    ) as fout, tqdm(total=len(lines)) as pbar:
        for line in lines:
            line = eval(line)
            score = call_api(prompt(line["question"], line["ctx"]))
            line["score"] = score
            fout.write(line)
            pbar.set_postfix_str(f"score: {score}")
            pbar.update(1)


def main():
    with open(
        "/home/wzc2022/dgt_workspace/LLM-Knowledge-alignment-dgt/data/roberta_data/part_qa_pair.jsonl",
        "r",
    ) as fin:
        lines = fin.readlines()
        n = len(lines) // 5
        threading.Thread(
            target=ask_part, name="wzcThread-1", args=(lines[:n], 1)
        ).start()
        threading.Thread(
            target=ask_part, name="wzcThread-2", args=(lines[n : n * 2], 2)
        ).start()
        threading.Thread(
            target=ask_part, name="wzcThread-3", args=(lines[n * 2 : n * 3], 3)
        ).start()
        threading.Thread(
            target=ask_part, name="wzcThread-4", args=(lines[n * 3 : n * 4], 4)
        ).start()
        # threading.Thread(
        #     target=ask_part, name="wzcThread-5", args=(lines[n * 4 :], 5)
        # ).start()


if __name__ == "__main__":
    main()
