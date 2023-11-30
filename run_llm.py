import os
from tqdm import tqdm
import json
import logging
import argparse
from utils.utils import load_source
from utils.llm import get_llm_result
from utils.prompt import get_prompt
from utils.utils import deal_answer, deal_judge, deal_post, str2paras
from transformers import AutoTokenizer, AutoModel
import openai
import time
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import torch
from tqdm import trange


ra_dict = {
    "none": "none",
    "sparse": {"sparse_ctxs": 10},
    "dense": {"dense_ctxs": 10},
    "chatgpt": {"gen_ctxs": 100},
    "sparse+dense": {"dense_ctxs": 5, "sparse_ctxs": 5},
    "gold": {"gold_ctxs": 10},
    "strong": {"strong_ctxs": 10},
    "weak": {"weak_ctxs": 10},
    "rand": {"rand_ctxs": 10},
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/source/nq.json")
    parser.add_argument("--usechat", action="store_true")
    parser.add_argument("--rank", action="store_true")
    parser.add_argument(
        "--type", type=str, choices=["qa", "prior", "post", "generate"], default="qa"
    )
    parser.add_argument("--gpu", type=str, choices=["0", "1"], default="0")
    parser.add_argument(
        "--model",
        type=str,
        choices=["chatglm", "llama", "chatgpt", "llama2"],
        default="llama",
    )
    parser.add_argument("--ra", type=str, default="none", choices=ra_dict.keys())
    parser.add_argument("--outfile", type=str, default="data/qa/chatgpt-nq-none.json")

    args = parser.parse_args()

    if args.type == "generate":
        assert (
            args.usechat and args.ra == "none"
        ), "You should use ChatGPT with no supporting documents to generate."
    args.ra = ra_dict[args.ra]

    return args


def get_llama_result(prompt, chat, sample, deal_type, model, tokenizer, args):
    # 处理访问频率过高的情况
    def get_res(prompt, chat=True, gen=False):
        # while True:
        encoded_inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = encoded_inputs["input_ids"].to(device)
        attention_mask = encoded_inputs["attention_mask"].to(device)

        if input_ids.shape[1] > 100:
            input_ids = input_ids[:, -100:]
            attention_mask = attention_mask[:, -100:]

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                temperature=0.8,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=True,
                # repetition_penalty=2.5,
                # length_penalty=1.0,
                early_stopping=True,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        res = [output]
        # import pdb
        # pdb.set_trace()

        return res

    def request_process(prompt, chat, sample, deal_type):
        gen = deal_type == "generate"
        res = get_res(prompt, chat=chat, gen=gen)
        prediction = None
        prediction = res[0] if res is not None else None
        # import pdb
        # pdb.set_trace()
        if deal_type == "post":
            sample["post_prompt"] = prompt
            sample["Post"] = prediction
            sample["Post_Giveup"], sample["Post_True"] = deal_post(prediction)
        elif deal_type == "qa":
            sample["qa_prompt"] = prompt
            sample["Prediction"] = prediction
            sample["EM"], sample["F1"] = deal_answer(prediction, sample["reference"])
        elif deal_type == "prior":
            sample["prior_prompt"] = prompt
            sample["Prior"] = prediction
            sample["Giveup"] = deal_judge(prediction)
        elif deal_type == "generate":
            sample["gen_prompt"] = prompt
            sample["gen_response"] = prediction
            sample["gen_ctxs"] = str2paras(prediction)
        return sample

    device = f"cuda:{args.gpu}"
    return request_process(prompt, chat, sample, deal_type)


def get_chatglm_result(prompt, chat, sample, deal_type, model, tokenizer):
    # 处理访问频率过高的情况
    def get_res(prompt, chat=True, gen=False):
        # while True:

        response, history = model.chat(tokenizer, prompt, history=[])
        # import pdb
        # pdb.set_trace()

        return [response]

    def request_process(prompt, chat, sample, deal_type):
        gen = deal_type == "generate"
        res = get_res(prompt, chat=chat, gen=gen)
        prediction = None
        prediction = res[0] if res is not None else None
        # import pdb
        # pdb.set_trace()
        if deal_type == "post":
            sample["post_prompt"] = prompt
            sample["Post"] = prediction
            sample["Post_Giveup"], sample["Post_True"] = deal_post(prediction)
        elif deal_type == "qa":
            sample["qa_prompt"] = prompt
            sample["Prediction"] = prediction
            sample["EM"], sample["F1"] = deal_answer(prediction, sample["reference"])
        elif deal_type == "prior":
            sample["prior_prompt"] = prompt
            sample["Prior"] = prediction
            sample["Giveup"] = deal_judge(prediction)
        elif deal_type == "generate":
            sample["gen_prompt"] = prompt
            sample["gen_response"] = prediction
            sample["gen_ctxs"] = str2paras(prediction)
        return sample

    # device = f"cuda:{args.gpu}"
    return request_process(prompt, chat, sample, deal_type)


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
# calculate_average_score("/home/zjx2022/dgt_workspace/LLM-Knowledge-alignment-dgt/data/qa/nq-sparse-qa.json")


def main():
    args = get_args()
    begin = 0
    if os.path.exists(args.outfile):
        outfile = open(args.outfile, "r", encoding="utf-8")
        for line in outfile.readlines():
            if line != "":
                begin += 1
        outfile.close()
        outfile = open(args.outfile, "a", encoding="utf-8")
    else:
        outfile = open(args.outfile, "w", encoding="utf-8")

    all_data = load_source(args.source)
    num_output = 0

    # llama
    device = f"cuda:{args.gpu}"

    print("training on " + device)

    if args.model == "llama":
        print("-------------------using llama as model--------------------------")

        path = "/home/sxs2022/PretrainedModel/llama-7b-hf"
        model = transformers.AutoModelForCausalLM.from_pretrained(path)

        tokenizer = LlamaTokenizer.from_pretrained(path)
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model = model.to(device)

    elif args.model == "llama2":
        print("-------------------using llama2 as model--------------------------")

        path = "/home/sxs2022/PretrainedModel/Llama-2-7b-chat-hf"
        model = transformers.AutoModelForCausalLM.from_pretrained(path)

        tokenizer = LlamaTokenizer.from_pretrained(path)
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model = model.to(device)

    elif args.model == "chatglm":
        print("-------------------using chatglm2 as model--------------------------")
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/sxs2022/PretrainedModel/ChatGLM2-6B", trust_remote_code=True
        )
        model = (
            AutoModel.from_pretrained(
                "/home/sxs2022/PretrainedModel/ChatGLM2-6B", trust_remote_code=True
            )
            .half()
            .cuda()
        )
        model = model.to(device)
        model = model.eval()
        # response, history = model.chat(tokenizer, "你好", history=[])
        # print(response)

    try:
        for sample in tqdm(all_data[begin:], desc="Filename: %s" % args.outfile):
            prompt = get_prompt(sample, args)
            # import pdb
            # pdb.set_trace()
            # sample = get_llm_result(prompt, args.usechat, sample, args.type)
            if args.model == "llama" or args.model == "llama2":
                sample = get_llama_result(
                    prompt, args.usechat, sample, args.type, model, tokenizer, args
                )
            if args.model == "chatglm":
                sample = get_chatglm_result(
                    prompt, args.usechat, sample, args.type, model, tokenizer
                )

            outfile.write(json.dumps(sample) + "\n")
            num_output += 1
    except Exception as e:
        logging.exception(e)

    finally:
        print(args.outfile, " has output %d line(s)." % num_output)
        print(calculate_average_score(args.outfile))
        outfile.close()


if __name__ == "__main__":
    main()
