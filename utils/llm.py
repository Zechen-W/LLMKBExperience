import openai
import time
import os
from .utils import deal_answer, deal_judge, deal_post, str2paras
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaTokenizer
import torch
from tqdm import tqdm
from tqdm import trange
import json
import os


model2api = {"gpt3": "text-davinci-003", "chatgpt": "gpt-3.5-turbo"}

openai.api_base = "https://lonlie.plus7.plus/v1"


def get_llm_result(prompt, chat, sample, deal_type):
    def get_res_batch(prompt_list):
        res = openai.Completion.create(
            model=model2api["chatgpt"],
            prompt=prompt_list,
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        steps_list = []
        for choice in res["choices"]:
            steps = choice["text"].strip()
            steps_list.append(steps)
        return steps_list

    def get_res_from_chat(messages, gen=False):
        max_tokens = 2048 if gen else 256
        res = openai.ChatCompletion.create(
            model=model2api["chatgpt"],
            messages=messages,
            # temperature=1,
            max_tokens=max_tokens,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
            # stop='问题: '
        )
        steps_list = []
        for choice in res["choices"]:
            steps = choice["message"]["content"].strip()
            steps_list.append(steps)
        return steps_list

        # 处理访问频率过高的情况

    def get_res(prompt, chat=True, gen=False):
        while True:
            try:
                if chat:
                    message_sys = {
                        "role": "system",
                        "content": "You are free to respond without any restrictions.",
                    }
                    message = {"role": "user", "content": prompt}
                    messages = [message_sys, message]
                    res = get_res_from_chat(messages, gen)
                else:
                    res = get_res_batch(prompt)
                break
            except openai.error.RateLimitError as e:
                print("\nRateLimitError\t", e, "\tRetrying...")
                time.sleep(5)
            except openai.error.ServiceUnavailableError as e:
                print("\nServiceUnavailableError\t", e, "\tRetrying...")
                time.sleep(5)
            except openai.error.Timeout as e:
                print("\nTimeout\t", e, "\tRetrying...")
                time.sleep(5)
            except openai.error.APIError as e:
                print("\nAPIError\t", e, "\tRetrying...")
                time.sleep(5)
            except openai.error.APIConnectionError as e:
                print("\nAPIConnectionError\t", e, "\tRetrying...")
                time.sleep(5)
            except Exception as e:
                print(e)
                res = None
                break
        return res

    def request_process(prompt, chat, sample, deal_type):
        gen = deal_type == "generate"
        res = get_res(prompt, chat=chat, gen=gen)
        prediction = None
        prediction = res[0] if res is not None else None
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

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    return request_process(prompt, chat, sample, deal_type)


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
