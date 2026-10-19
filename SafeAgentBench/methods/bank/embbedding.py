import json
import csv
import os
import numpy as np
import requests
import json


url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/embeddings"
Skey = os.environ["OPENAI_API_KEY"]

# 存储所有嵌入的列表
safe_embeddings = []
unsafe_embeddings = []
# 存储对应的文件信息
file_info = []

filename = "../../dataset/combined_data.json"

# data = []
# with open(filename, 'r') as file:
#     for line in file:
#         # 解析每一行 JSON 对象并添加到列表中
#         data.append(json.loads(line))
data = json.load(open(filename, 'r'))

for idx, example in enumerate(data):
# 提取intent字段
    safe = example['safe_task']['instruction']
    print(f"处理文件 {idx}...")
    payload = json.dumps({
        "model": "text-embedding-ada-002",
        "input": f"{safe}"
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {Skey}',
    }
    for _ in range(3):
        response = requests.request("POST", url, headers=headers, data=payload)
        if response.status_code == 200:
            break
    data = response.json()
    safe_embedding = data['data'][0]['embedding']
    if safe_embedding:
        # 存储嵌入和文件信息
        safe_embeddings.append(safe_embedding)

    unsafe = example['unsafe_task']['instruction']
    print(f"处理文件 {idx}...")
    payload = json.dumps({
        "model": "text-embedding-ada-002",
        "input": f"{unsafe}"
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {Skey}',
    }
    for _ in range(3):
        response = requests.request("POST", url, headers=headers, data=payload)
        if response.status_code == 200:
            break
    data = response.json()
    unsafe_embedding = data['data'][0]['embedding']
    if unsafe_embedding:
        # 存储嵌入和文件信息
        unsafe_embeddings.append(unsafe_embedding)

# 将嵌入保存到CSV文件
if safe_embedding:
    output_filename = "safe_embeddings.csv"
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for embedding in safe_embeddings:
            writer.writerow(embedding)
    
if unsafe_embedding:
    output_filename = "unsafe_embedding.csv"
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for embedding in unsafe_embeddings:
            writer.writerow(embedding)
    
