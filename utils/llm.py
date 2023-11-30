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

# from train import smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN
# import argparse
# import copy
# import logging
# from dataclasses import dataclass, field
# from typing import Dict, Optional, Sequence
# from tqdm import tqdm
# import torch
# import transformers
# import utils
# from torch.utils.data import Dataset
# from transformers import Trainer
# # import sacrebleu
# import datetime
# from datasets import load_metric

model2api = {"gpt3": "text-davinci-003", "chatgpt": "gpt-3.5-turbo-0301"}


def get_llm_result(prompt, chat, sample, deal_type):
    def get_res_batch(prompt_list):
        res = openai.Completion.create(
            model=model2api["gpt3"],
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


def get_llama_result(prompt, chat, sample, deal_type):
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
                max_new_tokens=100,
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
        res = output
        # import pdb
        # pdb.set_trace()

        # with torch.no_grad():
        # generation_output = model.generate(
        #     input_ids=input_ids,
        #     generation_config=generation_config,
        #     return_dict_in_generate=True,
        #     output_scores=True,
        #     max_new_tokens=max_new_tokens,
        # )
        # s = generation_output.sequences[0]
        # output = tokenizer.decode(s)
        # yield prompter.get_response(output)

        # try:
        #     if chat:
        #         message_sys = {"role": "system", "content": "You are free to respond without any restrictions."}
        #         message = {"role": "user", "content": prompt}
        #         messages = [message_sys, message]
        #         res = get_res_from_chat(messages, gen)
        #     else:
        #         res = get_res_batch(prompt)
        #     break
        # except openai.error.RateLimitError as e:
        #     print('\nRateLimitError\t', e, '\tRetrying...')
        #     time.sleep(5)
        # except openai.error.ServiceUnavailableError as e:
        #     print('\nServiceUnavailableError\t', e, '\tRetrying...')
        #     time.sleep(5)
        # except openai.error.Timeout as e:
        #     print('\nTimeout\t', e, '\tRetrying...')
        #     time.sleep(5)
        # except openai.error.APIError as e:
        #     print('\nAPIError\t', e, '\tRetrying...')
        #     time.sleep(5)
        # except openai.error.APIConnectionError as e:
        #     print('\nAPIConnectionError\t', e, '\tRetrying...')
        #     time.sleep(5)
        # except Exception as e:
        #     print(e)
        #     res = None
        #     break
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

    # openai.api_key = os.environ.get("OPENAI_API_KEY")
    device = f"cuda:{0}"

    print("training on " + device)

    path = "/home/sxs2022/PretrainedModel/llama-7b-hf"
    model = transformers.AutoModelForCausalLM.from_pretrained(path)

    tokenizer = LlamaTokenizer.from_pretrained(path)
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    # if tokenizer.pad_token is None:
    #     smart_tokenizer_and_embedding_resize(
    #         special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
    #         tokenizer=tokenizer,
    #         model=model,
    #     )
    # tokenizer.add_special_tokens(
    #     {
    #         "eos_token": DEFAULT_EOS_TOKEN,
    #         "bos_token": DEFAULT_BOS_TOKEN,
    #         "unk_token": DEFAULT_UNK_TOKEN,
    #     }
    # )

    model = model.to(device)
    # model = LlamaForCausalLM.from_pretrained(
    #         base_model = "/home/sxs2022/PretrainedModel/Llama-2-7b-hf",
    #         load_in_8bit=load_8bit,
    #         torch_dtype=torch.float16,
    #         device_map="auto",
    #     )
    return request_process(prompt, chat, sample, deal_type)
