import numpy as np
from transformers import AutoTokenizer, AutoConfig
import torch
from train_roberta import RobertaCLS

prompt_dict = {
    "qa": {
        "none": "Answer the following question based on your internal knowledge with one or few words.\nQuestion: {question}{paras}{prediction}",
        "ra": "Given the following information: \n{paras}\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source.\nQuestion: {question}{prediction}",
        "tail": "\nAnswer: ",
    },
    "prior": {
        "none": 'Are you sure to accurately answer the following question based on your internal knowledge, if yes, you should give a short answer with one or few words, if no, you should answer "Unknown"\nQuestion: {question}{paras}{prediction}',
        "ra": 'Given the following information: \n{paras}\nCan you answer the following question based on the given information or your internal knowledge, if yes, you should give a short answer with one or few words, if no, you should answer "Unknown".\nQuestion: {question}{prediction}',
        "tail": "\nAnswer: ",
    },
    "post": {
        "none": 'Can you judge if the following answer about the question is correct based on your internal knowledge, if yes, you should answer True or False, if no, you should answer "Unknown".\nQuestion: {question}{paras}\nAnswer: {prediction}',
        "ra": 'Given the following information: \n{paras}\nCan you judge the if the following answer about the question is correct based on the given information or your internal knowledge, if yes, you should answer True or False, if no, you should answer "Unknown".\nQuestion: {question}\nAnswer: {prediction}',
        "tail": "\nJudgement is: ",
    },
    "generate": {
        "none": 'I want you to act as a Wikipedia page. I will give you a question, and you will provide related passages in the format of a Wikipedia page which contains 10 paragraphs split by "\n\n". Your summary should be informative and factual, covering the key phrases that could answer the following question.\nQuestion: {question}{paras}{prediction}',
        "ra": "",
        "tail": "",
    },
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def roberta_rank(question, docs, tokenizer, model):
    X = tokenizer(
        [question] * len(docs),
        docs,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    pred = model(X).argmax(1)

    # 使用NumPy.argsort按照权重从大到小排序docs
    sorted_docs_indices = np.argsort(pred)[::-1]

    # 根据排序的索引重排docs列表
    sorted_docs = [docs[i] for i in sorted_docs_indices]

    # 输出排序后的docs

    return sorted_docs

    # question : str 'what was the result of the revolt of 1857'
    # docs : list ["23123214","ntest to complete Polk's term, and"]


def get_prompt(sample, args):
    paras = ""
    prompt = prompt_dict[args.type]["none"]
    # roberta rank时用
    if args.rank:
        print("---------------loading Roberta for ranking----------------")
        checkpoint = "/home/wzc2022/dgt_workspace/LLM-Knowledge-alignment-dgt/roberta_weights/roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        config = AutoConfig.from_pretrained(checkpoint)
        model = RobertaCLS(config).to(device)
        model.load_state_dict(torch.load(args.roberta_weight_path)).to(device)

    if args.ra != "none":
        ra_dict = args.ra
        i = 0
        doc = []

        for k, v in ra_dict.items():
            v = min(v, len(sample[k]))
            if args.rank:
                sample[k] = roberta_rank(
                    sample["question"], sample[k], tokenizer, model
                )

            for j in range(v):
                doc.append(("Passage-%d" % i) + sample[k][j])
                # import pdb
                # pdb.set_trace()
                i += 1
        paras = "\n".join(doc)
        prompt = prompt_dict[args.type]["ra"]
    tail = prompt_dict[args.type]["tail"] if not args.usechat else ""
    prediction = sample["Prediction"] if args.type == "post" else ""
    prompt = (
        prompt.format(question=sample["question"], paras=paras, prediction=prediction)
        + tail
    )
    return prompt
