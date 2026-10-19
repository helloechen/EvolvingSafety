import argparse
import base64
import io
import json
import os
import sys
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Thread
import random

import numpy as np
from PIL import Image
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='Qwen3-8B')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--sample_size', type=int, default=None,
                    help='Optional subset size; default uses all collected benign trajectories')
args = parser.parse_args()
model_name = args.model_name

if 'Qwen3' in model_name:
    agent_client = OpenAI(
        base_url=os.getenv("EVAL_BASE_URL"),
    )
    gpt_model = 'Qwen/' + model_name
    model_tag = model_name.split('-')[-1]
else:
    agent_client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    gpt_model = model_name
    model_tag = model_name.split('-')[0]

with open(f"bank/awm_{model_tag}_bank.json") as file:
    all_data = json.load(file)
if not all_data:
    raise ValueError("The offline trajectory bank is empty")
all_data = sorted(all_data, key=lambda item: item['idx'])
if args.sample_size is None:
    final_data = all_data
elif 0 < args.sample_size <= len(all_data):
    final_data = random.Random(args.seed).sample(all_data, args.sample_size)
else:
    raise ValueError("sample_size must be between 1 and the number of collected trajectories")

INSTRUCTION = open("workflow/instruction.txt").read()
prompt = ''

for idx, data in enumerate(final_data):
    prompt += f"================= Task {idx+1}: =================\n" + \
        f"The goal is: {data['safe_task']['instruction']}" + \
        f"The Reference Steps that can complete the task are: {data['safe_task']['step']}" + \
        f"The low level plan that the agent finally used to accomplish the task is: {data['safe_task']['low_level_plan']}" +  \
        f"The evaluation result of the plan is: **{'SUCCEED' if data['safe_task']['llm_success'] else 'FAILED'}**.\nThe detailed evaluation analysis is: {data['safe_task']['detailed_evaluation_response']}" + '\n\n' 

prompt = INSTRUCTION + '\n\n' + prompt

messages = [{"role": "user", "content": prompt}]
for _ in range(3):
    response = agent_client.chat.completions.create(
        model=gpt_model, 
        messages=messages,
        temperature=0.1,
        max_tokens=2048,
        top_p=1.0,
        extra_body={
            "top_k": 0,
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )
    if response.choices[0].message.content.strip() != "":
        break

workflow = response.choices[0].message.content
with open(f"workflow/workflow_{model_tag}.txt", 'w') as f:
    f.write(workflow)
    



