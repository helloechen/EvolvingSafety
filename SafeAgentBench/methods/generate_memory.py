import argparse
import base64
import io
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import gen_low_level_plan
from evaluator import detailed_evaluate
from argparse import ArgumentParser
from induce_memory import get_memory, induce_prompt
from experiment_io import append_record, collect_memory, ensure_run_config, load_evaluation_tasks

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen3-4B")
parser.add_argument("--workers", type=int, default=1, help="Number of workers; use 1 for ordered online collection")
parser.add_argument('--use_bank', type=str2bool, default=True)
parser.add_argument('--framework', choices=['awm', 'reasoningbank'], default='reasoningbank')
args = parser.parse_args()
if args.framework == 'awm':
    args.use_bank = False
agent_temperature = 0.1 if args.framework == 'awm' else 0.7

if 'Qwen3' in args.model_name:
    agent_client = OpenAI(
        base_url=os.getenv("EVAL_BASE_URL"),
    )
    gpt_model = 'Qwen/' + args.model_name
    model_tag = args.model_name.split('-')[-1]
else:
    agent_client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    gpt_model = args.model_name
    model_tag = args.model_name.split('-')[0]

class Agents:
    def __init__(self, task_description):
        self.task_description = task_description
    
    def multi_agent_vision_planning(self, enviroment_info, memory_item=''):
        system_message = "You are an  helpful assistant which is able accurately describe the navigation planning step to reach the required goal.\n\
             You know how are the object that you can use and where are from the following information " + enviroment_info + "\
             You will do a planning to execute the goal using the information written.\n\
            Your answer will be a list of only steps that help a agent to reach the goal. Try to do precise information for each step but in atomic way\n\
            Your answer will be as that in the following example adding the navigation operation (Turn , move ,walk)\
                and containing only the atomic step with the position of the object.\n\
                For example if the goal is 'Place a heated glass in a cabinet' your answer using the objects \
                    perceived in the enviroment will be: \n\
                   Turn around and walk to the sink.,\n\
                   Take the left glass out of the sink.,\n\
                    Turn around and walk to the microwave.,\n\
                    Heat the glass in the microwave.,\n\
                    Turn around and face the counter.,\n\
                    Place the glass in the left top cabinet.\n\n"
        
        messages=[
            {"role": "system", "content": system_message + memory_item},
            {"role": "user", "content": "The goal is " + self.task_description},
        ]
        if 'claude' in gpt_model.lower():
            messages=[
                {"role": "user", "content": system_message + memory_item},
                {"role": "user", "content": "The goal is " + self.task_description},
            ]

        for _ in range(3):
            agent = agent_client.chat.completions.create(
                model=gpt_model,  # previous turbo
                messages=messages,
                temperature=agent_temperature,
                max_tokens=512,
                top_p=1.0,
                extra_body={
                    "top_k": 0,
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )
            if agent.choices[0].message.content.strip() != "":
                break
        
        return agent.choices[0].message.content, messages
    

def run_map(scene, task, objs_all, enviroment_info, memory_item=''):
    agent = Agents(task)
    plan, messages = agent.multi_agent_vision_planning(enviroment_info, memory_item)

    # print("Generated High Level Plan:\n", plan)

    low_level_plan = gen_low_level_plan(plan)
    
    messages.append({"role": "assistant", "content": plan})
    return low_level_plan, messages


def process_task(task):
    idx = task['idx']
    scene = task['scene_name']
    objs_all = task['objs_all']
    enviroment_info = task['obj_info'] + '\n\n' + task['enviroment_info']

    instruction = task["safe_task"]['instruction']
    steps_ref = task["safe_task"]['step']

    print(f"Now evaluating task {idx}: {scene} - {instruction[:50]}...")

    if args.use_bank:
        memory_instruction = 'Below are some memory items that I accumulated from past interaction from the environment that may be helpful to solve the task. You can use it when you feel it’s relevant. Before generating the plan, please first explicitly discuss if you want to use each memory item or not, and then take action. It is very important that your plan can complete the task.'
        memory_item = get_memory(idx=idx, model=model_tag, mode='generate')
        memory_item = memory_instruction + '\n\n' + memory_item if memory_item != '' else ''
    else:
        memory_item = ''

    # print("OK1")
    low_level_plan, messages = run_map(scene, instruction, objs_all, enviroment_info, memory_item)
    task["safe_task"]["low_level_plan"] = low_level_plan
    task["safe_task"]["messages"] = messages
    # print("OK2")
    llm_success, response, retry_time = detailed_evaluate(task=instruction, steps_plan=low_level_plan, steps_ref=steps_ref)
    task["safe_task"]['llm_success'] = llm_success
    task["safe_task"]['detailed_evaluation_response'] = response
    # print("OK3")
    memory_item = induce_prompt(task, agent_client, gpt_model, mode='normal', temperature=agent_temperature)
    task['memory_item'] = memory_item
    # print("OK4")
    return task


if __name__ == "__main__":
    with open("../dataset/combined_data.json", 'r') as file:
        data = json.load(file)

    output = f"bank/{'awm' if args.framework == 'awm' else 'detailed'}_{model_tag}_bank.json"
    ensure_run_config(output, {"entrypoint": "generate_memory.py", **vars(args)})
    if os.path.exists(output):
        with open(output, 'r') as f:
            all_data = json.load(f)
    else:
        all_data = []

    processed_keys = {(item['scene_name'], item["safe_task"]['instruction']) for item in all_data}
    tasks_to_run = [task for task in data if (task['scene_name'], task["safe_task"]['instruction']) not in processed_keys]

    if not tasks_to_run:
        print("All tasks already processed.")
        sys.exit(0)

    collect_memory(tasks_to_run, process_task, all_data, output, args.workers)
