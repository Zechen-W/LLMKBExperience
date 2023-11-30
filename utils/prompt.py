from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np

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


def encode_text(text):
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    return input_ids


def roberta_rank(question, docs):
    # TODO ROBERTa 排序q 与 docs

    encoded_question = encode_text(question)
    encoded_docs = [encode_text(doc) for doc in docs]

    question_embeddings = model(encoded_question)[0][:, 0, :]
    doc_embeddings = [model(encoded_doc)[0][:, 0, :] for encoded_doc in encoded_docs]

    sims = []
    for doc_embedding in doc_embeddings:
        sim = np.inner(
            question_embeddings.detach().numpy(), doc_embedding.detach().numpy()
        )
        sims.append(sim)

    sorted_docs = [doc for _, doc in sorted(zip(sims, docs), reverse=True)]
    return sorted_docs

    # question : str 'what was the result of the revolt of 1857'
    # docs : list ["23123214","ntest to complete Polk's term, and"]


def get_prompt(sample, args):
    paras = ""
    prompt = prompt_dict[args.type]["none"]
    # roberta rank时用
    if args.rank:
        print("---------------loading Roberta for ranking----------------")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = TFRobertaModel.from_pretrained("roberta-base")
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
    # import pdb
    # pdb.set_trace()
    return prompt
