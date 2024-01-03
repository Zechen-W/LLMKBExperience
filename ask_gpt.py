import jsonlines
from sympy import false
from connect_openai import CallFailException, call_api
from tqdm import tqdm
import threading
from utils.prompt import prompt_dict
from utils.utils import deal_answer


def prompt(q, ctx):
    return f"""As a professional data annotator, I will give you a question and a context below. Please judge whether to answer the given question, in addition to the knowledge you already have, you need the given context as a supplement to answer the question. You scored on a scale of 0 to 5, where 0 means you can answer the question using only your knowledge and no context at all, and 5 means you can't answer the question using your current knowledge without context.\n

question: {q}
context: {ctx}\n

Note that your answer should contain only a number and nothing else.
"""


def ask_part(lines, n_part):
    with jsonlines.open(
        f"/home/wzc2022/dgt_workspace/LLM-Knowledge-alignment-dgt/data/roberta_data/part_qa_pair_compare_{n_part}.jsonl",
        "w",
    ) as fout, tqdm(total=len(lines)) as pbar:
        last_q_id = None
        for line in lines:
            line = eval(line)
            # 如果更新了问题，不检索直接问
            if line["q_id"] != last_q_id:
                prompt = prompt_dict["qa"]["none"].format(
                    question=line["question"], paras="", prediction=""
                )
                # 不带passage直接问，一定要调用成功
                success = False
                while not success:
                    try:
                        before_answer = call_api(prompt)
                        success = True
                    except CallFailException as e:
                        print(e.message)
                before_em, before_f1 = deal_answer(before_answer, line["reference"])
            line["before"] = {
                "answer": before_answer,
                "em": before_em,
                "f1": before_f1,
            }
            # 带检索问，可以不成功，不成功就丢弃
            prompt = prompt_dict["qa"]["ra"].format(
                question=line["question"], paras=line["ctx"], prediction=""
            )
            try:
                answer = call_api(prompt)
                em, f1 = deal_answer(answer, line["reference"])
                line["after"] = {"answer": answer, "em": em, "f1": f1}

                # 0：错错 1：错对 2：对错 3：对对
                category = (int(before_f1 > 0.5) << 1) + int(f1 > 0.5)
                line["category"] = category
                fout.write(line)
                pbar.set_postfix_str(f"category: {category}")
            except CallFailException as e:
                print(e.message)
                print("达到最大重试次数，丢弃此数据")
                continue
            finally:
                last_q_id = line["q_id"]

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
        threading.Thread(
            target=ask_part, name="wzcThread-5", args=(lines[n * 4 :], 5)
        ).start()


if __name__ == "__main__":
    main()
