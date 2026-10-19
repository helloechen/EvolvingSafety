import os
import json
import random
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def induce_prompt(task, agent_client, gpt_model, mode='normal', verbose=False, temperature=0.7):
    """Call gpt model to generate memory_item.""" 
    if mode == 'normal':
        SUCCEED_PROMPT = open("bank/succeed_prompt.txt").read()
        FAILED_PROMPT = open("bank/failed_prompt.txt").read()
        is_success = task["safe_task"]['llm_success']

        enviroment_info = f"Here is the enviroment info where the agent accomplish the task:\n{task['obj_info']}\n\n{task['enviroment_info']}"
        instruction = f"The goal is: {task['safe_task']['instruction']}"
        ref = f"The Reference Steps that can complete the task are: {task['safe_task']['step']}"
        plan = f"The low level plan that the agent finally used to accomplish the task is: {task['safe_task']['low_level_plan']}"
        evaluation = f"The evaluation result of the plan is: **{'SUCCEED' if is_success else 'FAILED'}**.\nThe detailed evaluation analysis is: {task['safe_task']['detailed_evaluation_response']}"

        if is_success:
            ipt = '\n\n'.join([SUCCEED_PROMPT, enviroment_info, instruction, ref, plan, evaluation])
        else:
            ipt = '\n\n'.join([FAILED_PROMPT, enviroment_info, instruction, ref, plan, evaluation])
        if verbose: print("Prompt:\n", ipt, '\n\n')
    elif mode in ["refusal", "execution", "mix", "safety_ref"]:
        SUCCEED_PROMPT = open("safety/prompt/success_prompt.txt").read()
        FAILED_PROMPT = open("safety/prompt/failed_prompt.txt").read()
        is_success = task["safe_task"]['llm_success']

        enviroment_info = f"Here is the enviroment info where the agent accomplish the task:\n{task['obj_info']}\n\n{task['enviroment_info']}"
        instruction = f"The goal is: {task['safe_task']['instruction']}"
        ref = f"The Reference Steps that can complete the task are: {task['safe_task']['step']}"
        plan = f"The low level plan that the agent finally used to accomplish the task is: {task['safe_task']['low_level_plan']}"
        analysis = f"The agent's raw analysis of the task is: {task['safe_task']['messages'][-1]['content']}"
        evaluation = f"The evaluation result of the plan is: **{'SUCCEED' if is_success else 'FAILED'}**.\nThe detailed evaluation analysis is: {task['safe_task']['detailed_evaluation_response']}"

        if is_success:
            ipt = '\n\n'.join([SUCCEED_PROMPT, enviroment_info, instruction, ref, plan, analysis, evaluation])
        else:
            ipt = '\n\n'.join([FAILED_PROMPT, enviroment_info, instruction, ref, plan, analysis, evaluation])
        if verbose: print("Prompt:\n", ipt, '\n\n')
    elif mode == 'safety_ref':
        SUCCESS_REF = open("safety/success_ref.txt").read()
        FAILED_REF = open("safety/failed_ref.txt").read()
        is_success = task["safe_task"]['llm_success']

        enviroment_info = f"Here is the enviroment info where the agent accomplish the task:\n{task['obj_info']}\n\n{task['enviroment_info']}"
        instruction = f"The goal is: {task['safe_task']['instruction']}"
        ref = f"The Reference Steps that can complete the task are: {task['safe_task']['step']}"
        plan = f"The low level plan that the agent finally used to accomplish the task is: {task['safe_task']['low_level_plan']}"
        evaluation = f"The evaluation result of the plan is: **{'SUCCEED' if is_success else 'FAILED'}**.\nThe detailed evaluation analysis is: {task['safe_task']['detailed_evaluation_response']}"

        if is_success:
            ipt = '\n\n'.join([SUCCESS_REF, enviroment_info, instruction, ref, plan, evaluation])
        else:
            ipt = '\n\n'.join([FAILED_REF, enviroment_info, instruction, ref, plan, evaluation])
        if verbose: print("Prompt:\n", ipt, '\n\n')


    messages = [{"role": "system", "content": ipt}]
    if 'claude' in gpt_model.lower():
        messages = [{"role": "user", "content": ipt + '\n\nDO NOT generate anything other than memory_item!\n'}]
    for _ in range(3):
        response = agent_client.chat.completions.create(
            model=gpt_model, 
            messages=messages,
            temperature=temperature,
            max_tokens=2048,
            top_p=1.0,
            extra_body={
                "top_k": 0,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
        if response.choices[0].message.content.strip() != "":
            break
    
    memory_item = response.choices[0].message.content
    if verbose: print(memory_item)
    return memory_item


def get_memory(idx, model='8B', mode='generate', bank_count=10, return_multiple=False, safety_mode='execution', base_count=10) -> str:
    """
    返回最相似的一个task的memory_item
    """
    if mode == 'alf_detailed' or mode == 'alf_long':
        if bank_count == 0:
            print("Bank count is 0.")
            return ''
        # 1. 读取bank文件获取task_id列表alfworld_Qwen3-4B_LongSystemMessage
        bank_path = f"./bank/alfworld_Qwen3-{model}_LongSystemMessage.json"
        if not os.path.exists(bank_path):
            print(f"Bank file does not exist: {bank_path}")
            return ''
        with open(bank_path, 'r', encoding='utf-8') as f:
            bank = json.load(f)
        bank_data = []
        for item in bank:
            if item["task_idx"] < bank_count:
                bank_data.append(item)
        tid = idx
        task_ids = [int(item['task_idx']) for item in bank_data if 'task_idx' in item]
        if not task_ids:
            print("No current memory.")
            return ''

        # 2. 读取嵌入向量
        if mode == 'alf_detailed': 
            embeddings_df = pd.read_csv("./bank/detailed_embeddings.csv", header=None)
        else:
            embeddings_df = pd.read_csv("./bank/intent_embeddings.csv", header=None)
        alfworld_embeddings = pd.read_csv("./bank/alfworld.csv", header=None)
        # 检查tid有效性
        if tid >= len(embeddings_df) or tid < 0:
            print("Invalid current task id.")
            return ''
        # 获取目标向量和bank向量
        target_embedding = embeddings_df.iloc[tid].values.reshape(1, -1)
        # 过滤有效task_id
        valid_task_ids = [task_id for task_id in task_ids if task_id < len(alfworld_embeddings) and task_id >= 0]
        if not valid_task_ids:
            print("No valid task ids in bank.")
            return ''
        bank_embeddings = alfworld_embeddings.iloc[valid_task_ids].values

         # 3. 计算相似度
        similarities = cosine_similarity(target_embedding, bank_embeddings)
        most_similar_idx = np.argmax(similarities[0])

        similar_tid = valid_task_ids[most_similar_idx]
        memory_item = ''
        for item in bank_data:
            if int(item['task_idx']) == similar_tid:
                memory_item = item['memory_item']

        if memory_item == '':
            print("No memory item found for the most similar task.")
            print(f"Most similar task id: {similar_tid}")
        return memory_item

    elif mode == 'generate':
        # 1. 读取bank文件获取task_id列表
        bank_path = f"bank/detailed_{model}_bank.json"
        if not os.path.exists(bank_path):
            print("No current memory.")
            return ''
        with open(bank_path, 'r', encoding='utf-8') as f:
            bank_data = json.load(f)
        task_ids = list(set(int(item['idx']) for item in bank_data if 'idx' in item and int(item['idx']) != idx))
        if not task_ids:
            print("No current memory.")
            return ''

        # 2. 读取嵌入向量
        embeddings_df = pd.read_csv("bank/safe_embeddings.csv", header=None)
        # 检查tid有效性
        if idx >= len(embeddings_df) or idx < 0:
            print("Invalid current task id.")
            return ''
        # 获取目标向量和bank向量
        target_embedding = embeddings_df.iloc[idx].values.reshape(1, -1)
        # 过滤有效task_id
        valid_task_ids = [task_id for task_id in task_ids if task_id < len(embeddings_df) and task_id >= 0]
        if not valid_task_ids:
            print("No valid task ids in bank.")
            return ''
        bank_embeddings = embeddings_df.iloc[valid_task_ids].values

        # 3. 计算相似度
        similarities = cosine_similarity(target_embedding, bank_embeddings)
        most_similar_idx = np.argmax(similarities[0])

        similar_tid = valid_task_ids[most_similar_idx]
        memory_item = ''
        for item in bank_data:
            if int(item['idx']) == similar_tid:
                memory_item = item['memory_item']

        if memory_item == '':
            print("No memory item found for the most similar task.")
            print(f"Most similar task id: {similar_tid}")
        return memory_item

    elif mode == 'detailed' and not return_multiple:
        if bank_count == 0:
            print("Bank count is 0.")
            return ''
        # 1. 读取bank文件获取task_id列表alfworld_Qwen3-4B_LongSystemMessage
        bank_path = f"bank/detailed_{model}_bank.json"
        if not os.path.exists(bank_path):
            print(f"Bank file does not exist: {bank_path}")
            return ''
        with open(bank_path, 'r', encoding='utf-8') as f:
            bank = json.load(f)
        bank_data = []
        for item in bank:
            if item["idx"] < bank_count:
                bank_data.append(item)
        task_ids = [int(item['idx']) for item in bank_data if 'idx' in item]
        if not task_ids:
            print("No current memory.")
            return ''

        # 2. 读取嵌入向量
        unsafe_embeddings = pd.read_csv("./bank/unsafe_embeddings.csv", header=None)
        safe_embeddings = pd.read_csv("./bank/safe_embeddings.csv", header=None)
        # 检查tid有效性
        if idx >= len(unsafe_embeddings) or idx < 0:
            print("Invalid current task id.")
            return ''
        # 获取目标向量和bank向量
        target_embedding = unsafe_embeddings.iloc[idx].values.reshape(1, -1)
        # 过滤有效task_id
        valid_task_ids = [task_id for task_id in task_ids if task_id < len(safe_embeddings) and task_id >= 0]
        if not valid_task_ids:
            print("No valid task ids in bank.")
            return ''
        bank_embeddings = safe_embeddings.iloc[valid_task_ids].values

         # 3. 计算相似度
        similarities = cosine_similarity(target_embedding, bank_embeddings)
        most_similar_idx = np.argmax(similarities[0])

        similar_tid = valid_task_ids[most_similar_idx]
        memory_item = ''
        for item in bank_data:
            if int(item['idx']) == similar_tid:
                memory_item = item['memory_item']

        if memory_item == '':
            print("No memory item found for the most similar task.")
            print(f"Most similar task id: {similar_tid}")
        return memory_item

    elif mode == 'detailed' and return_multiple:
        if bank_count == 0:
            print("Bank count is 0.")
            return '', '', ''
        # 1. 读取bank文件获取task_id列表alfworld_Qwen3-4B_LongSystemMessage
        bank_path = f"bank/detailed_{model}_bank.json"
        if not os.path.exists(bank_path):
            print(f"Bank file does not exist: {bank_path}")
            return '', '', ''
        with open(bank_path, 'r', encoding='utf-8') as f:
            bank = json.load(f)
        bank_data = []
        for item in bank:
            if item["idx"] < bank_count:
                bank_data.append(item)
        task_ids = [int(item['idx']) for item in bank_data if 'idx' in item]
        if not task_ids:
            print("No current memory.")
            return '', '', ''

        # 2. 读取嵌入向量
        unsafe_embeddings = pd.read_csv("./bank/unsafe_embeddings.csv", header=None)
        safe_embeddings = pd.read_csv("./bank/safe_embeddings.csv", header=None)
        # 检查tid有效性
        if idx >= len(unsafe_embeddings) or idx < 0:
            print("Invalid current task id.")
            return '', '', ''
        # 获取目标向量和bank向量
        target_embedding = unsafe_embeddings.iloc[idx].values.reshape(1, -1)
        # 过滤有效task_id
        valid_task_ids = [task_id for task_id in task_ids if task_id < len(safe_embeddings) and task_id >= 0]
        if not valid_task_ids:
            print("No valid task ids in bank.")
            return '', '', ''
        bank_embeddings = safe_embeddings.iloc[valid_task_ids].values

         # 3. 计算相似度
        similarities = cosine_similarity(target_embedding, bank_embeddings)
        # 获取相似度的索引排序
        sorted_indices = np.argsort(similarities[0])
        most_similar_idx0 = sorted_indices[-1]
        most_similar_idx1 = sorted_indices[-2]
        most_similar_idx2 = sorted_indices[-3]

        similar_tid0 = valid_task_ids[most_similar_idx0]
        memory_item0 = ''
        for item in bank_data:
            if int(item['idx']) == similar_tid0:
                memory_item0 = item['memory_item']
        
        similar_tid1 = valid_task_ids[most_similar_idx1]
        memory_item1 = ''
        for item in bank_data:
            if int(item['idx']) == similar_tid1:
                memory_item1 = item['memory_item']

        similar_tid2 = valid_task_ids[most_similar_idx2]
        memory_item2 = ''
        for item in bank_data:
            if int(item['idx']) == similar_tid2:
                memory_item2 = item['memory_item']

        return memory_item0, memory_item1, memory_item2
    
    # elif mode == 'safety_eval':
    #     # 1. 读取bank文件获取task_id列表alfworld_Qwen3-4B_LongSystemMessage

    #     # if safety_mode == 'normal' or safety_mode == 'only_safe' or safety_mode == 'only_unsafe':
    #     #     bank_path = f"safety/{model}_bank.json"
    #     # elif safety_mode == 'mix':
    #     #     bank_path = f"safety/{model}_bank2.json"
    #     # elif safety_mode == 'combine':
    #     #     bank_path = f"safety/{model}_bank3.json"
    #     # elif safety_mode == 'combine2':
    #     #     bank_path = f"safety/{model}_bank5.json"
    #     # if not os.path.exists(bank_path):
    #     #     print(f"Bank file does not exist: {bank_path}")
    #     #     return ''
    #     if safety_mode != 'safety_ref':
    #         with open(f"safety/bank/{model}_{safety_mode}_base.json", 'r', encoding='utf-8') as f:
    #             base_bank = json.load(f)
    #             base_bank = base_bank[:base_count]  
    #         with open(f"safety/bank/{model}_{safety_mode}_bank.json", 'r', encoding='utf-8') as f:
    #             bank = json.load(f)
    #     else:
    #         base_bank = []
    #         if os.path.exists(f"safety/bank/ref/{model}_ref_bank.json"):
    #             with open(f"safety/bank/ref/{model}_ref_bank.json", 'r', encoding='utf-8') as f:
    #                 bank = json.load(f)
    #         else:
    #             print("No current memory.")
    #             return ''
        
    #     bank_data = []
    #     for item in bank:
    #         if safety_mode == 'only_safe':
    #             if item["idx"] < bank_count and item['unsafe_task']['llm_success'] == 0:
    #                 bank_data.append(item)
    #         elif safety_mode == 'only_unsafe':
    #             if item["idx"] < bank_count and item['unsafe_task']['llm_success'] == 1:
    #                 bank_data.append(item)
    #         else:
    #             if item["idx"] < bank_count:
    #                 bank_data.append(item)

    #     bank_data = base_bank + bank_data
    #     task_ids = [int(item['idx']) for item in bank_data if 'idx' in item]
    #     if not task_ids:
    #         print("No current memory.")
    #         return ''

    #     # 2. 读取嵌入向量
    #     unsafe_embeddings = pd.read_csv("./bank/unsafe_embeddings.csv", header=None)
    #     # 检查tid有效性
    #     if idx >= len(unsafe_embeddings) or idx < 0:
    #         print("Invalid current task id.")
    #         return ''
    #     # 获取目标向量和bank向量
    #     target_embedding = unsafe_embeddings.iloc[idx].values.reshape(1, -1)
    #     # 过滤有效task_id
    #     valid_task_ids = [task_id for task_id in task_ids if task_id < len(unsafe_embeddings) and task_id >= 0]
    #     if not valid_task_ids:
    #         print("No valid task ids in bank.")
    #         return ''
    #     bank_embeddings = unsafe_embeddings.iloc[valid_task_ids].values

    #      # 3. 计算相似度
    #     similarities = cosine_similarity(target_embedding, bank_embeddings)
    #     most_similar_idx = np.argmax(similarities[0])

    #     similar_tid = valid_task_ids[most_similar_idx]
    #     memory_item = ''
    #     for item in bank_data:
    #         if int(item['idx']) == similar_tid:
    #             memory_item = item['memory_item']

    #     if memory_item == '':
    #         print("No memory item found for the most similar task.")
    #         print(f"Most similar task id: {similar_tid}")
    #     return memory_item

    elif mode == 'safety':
        # 1. 读取bank文件获取task_id列表
        with open(f"safety/bank/base/{model}_{safety_mode}_base.json", 'r', encoding='utf-8') as f:
            base_bank = json.load(f)
        bank_path = f"safety/bank/{model}_{safety_mode}_bank.json"
        if os.path.exists(bank_path):
            with open(bank_path, 'r', encoding='utf-8') as f:
                bank_data = json.load(f)
        else:
            bank_data = []
        bank_data = base_bank + bank_data
        task_ids = list(set(int(item['idx']) for item in bank_data if 'idx' in item and int(item['idx']) != idx))
        
        # 2. 读取嵌入向量
        embeddings_df = pd.read_csv("bank/safe_embeddings.csv", header=None)
        # 检查tid有效性
        if idx >= len(embeddings_df) or idx < 0:
            print("Invalid current task id.")
            return ''
        # 获取目标向量和bank向量
        target_embedding = embeddings_df.iloc[idx].values.reshape(1, -1)
        # 过滤有效task_id
        valid_task_ids = [task_id for task_id in task_ids if task_id < len(embeddings_df) and task_id >= 0]
        if not valid_task_ids:
            print("No valid task ids in bank.")
            return ''
        bank_embeddings = embeddings_df.iloc[valid_task_ids].values

        # 3. 计算相似度
        similarities = cosine_similarity(target_embedding, bank_embeddings)
        # 获取相似度的索引排序
        sorted_indices = np.argsort(similarities[0])
        most_similar_idx0 = sorted_indices[-1]
        most_similar_idx1 = sorted_indices[-2]
        most_similar_idx2 = sorted_indices[-3]

        similar_tid0 = valid_task_ids[most_similar_idx0]
        memory_item0 = ''
        for item in bank_data:
            if int(item['idx']) == similar_tid0:
                memory_item0 = item['memory_item']
        similar_tid1 = valid_task_ids[most_similar_idx1]
        memory_item1 = ''
        for item in bank_data:
            if int(item['idx']) == similar_tid1:
                memory_item1 = item['memory_item']
        similar_tid2 = valid_task_ids[most_similar_idx2]
        memory_item2 = ''
        for item in bank_data:
            if int(item['idx']) == similar_tid2:
                memory_item2 = item['memory_item']
        return [memory_item0, memory_item1, memory_item2]

    elif mode == 'safety_eval':
        # 1. 读取bank文件获取task_id列表
        with open(f"safety/bank/base/{model}_{safety_mode}_base.json", 'r', encoding='utf-8') as f:
            base_bank = json.load(f)
            base_bank = base_bank[:base_count]  
        with open(f"safety/bank/{model}_{safety_mode}_bank.json", 'r', encoding='utf-8') as f:
            bank = json.load(f)
        bank_data = []
        for item in bank:
            if item["idx"] < bank_count:
                bank_data.append(item)
        bank_data = base_bank + bank_data
        task_ids = [int(item['idx']) for item in bank_data if 'idx' in item and int(item['idx']) != idx]
        
        if not task_ids:
            print("No current memory.")
            return ''

        # 2. 读取嵌入向量
        for type in ['unsafe', 'safe']:
            embeddings_df = pd.read_csv(f"bank/{type}_embeddings.csv", header=None)
            # 检查tid有效性
            if idx >= len(embeddings_df) or idx < 0:
                print("Invalid current task id.")
                return ''
            # 获取目标向量和bank向量
            target_embedding = embeddings_df.iloc[idx].values.reshape(1, -1)
            # 过滤有效task_id
            valid_task_ids = [task_id for task_id in task_ids if task_id < len(embeddings_df) and task_id >= 0]
            if not valid_task_ids:
                print("No valid task ids in bank.")
                return ''
            bank_embeddings = embeddings_df.iloc[valid_task_ids].values

            # 3. 计算相似度
            similarities = cosine_similarity(target_embedding, bank_embeddings)
            # 获取相似度的索引排序
            sorted_indices = np.argsort(similarities[0])
            most_similar_idx0 = sorted_indices[-1]
            most_similar_idx1 = sorted_indices[-2]
            most_similar_idx2 = sorted_indices[-3]

            similar_tid0 = valid_task_ids[most_similar_idx0]
            memory_item0 = ''
            for item in bank_data:
                if int(item['idx']) == similar_tid0:
                    memory_item0 = item['memory_item']
            similar_tid1 = valid_task_ids[most_similar_idx1]
            memory_item1 = ''
            for item in bank_data:
                if int(item['idx']) == similar_tid1:
                    memory_item1 = item['memory_item']
            similar_tid2 = valid_task_ids[most_similar_idx2]
            memory_item2 = ''
            for item in bank_data:
                if int(item['idx']) == similar_tid2:
                    memory_item2 = item['memory_item']
            if type == 'unsafe':
                unsafe_memory =  [memory_item0, memory_item1, memory_item2]
            else:
                safe_memory =  [memory_item0, memory_item1, memory_item2]

        return safe_memory, unsafe_memory
