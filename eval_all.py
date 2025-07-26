import random
import utils
import score
from pprint import pprint
import json
import os
from datetime import datetime
import spacy

data_ls = [
    
]
res_dict = {}
spacy_nlp = spacy.load("en_core_web_md")
def extract_entities(text):
    doc = spacy_nlp(text)
    entities = [(ent.text) for ent in doc.ents]
    return entities

import pandas as pd

def pr_dict(dict):
    rows = []
    for key, metrics in dict.items():
        model = "llama"
        method = key
        
        # 按行添加数据
        row = [model, method] + list(metrics.values())
        rows.append(row)
        
        
    # 构造DataFrame
    columns = ["Model", "Method"] + list(dict[list(dict.keys())[0]].keys())
    
    df = pd.DataFrame(rows, columns=columns)
    df.iloc[:, 2:] = df.iloc[:, 2:].map(lambda x: round(x * 100, 2) if isinstance(x, (int, float)) else x)


    df_sorted = df.sort_values(by=["Model", "Method"], ascending=[True, True])
    # 打印表格
    print(df_sorted)


def get_score(s):
    ### 从一个score类中，得到某个值
    return s['rougeL'].fmeasure

def save_data_to_json(data):
    """
    获取当前时间，并在指定路径下创建以当前时间为文件名的 JSON 文件，将 data 写入该文件。
    """
    # 获取当前时间，格式化为字符串
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 定义保存路径
    save_path = "./res_all"
    
    # 确保目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 构造文件路径
    file_path = os.path.join(save_path, f"{current_time}.json")
    
    # 将数据写入 JSON 文件
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    return file_path

def process_string(s):
    if s.startswith("Updated version:"):
        s = s[len("Updated version:"):].strip()
    elif s.startswith("Article:"):
        s = s[len("Article:"):].strip()
    
    if "Updated version:" in s:
        return s.split("Updated version:")[0].strip()
    if "Article:" in s:
        return s.split("Article:")[0].strip()
    
    if "Acticle:" in s:
        return s.split("Acticle:")[0].strip()
    
    return s.strip()


def process_item(item):
    item_score = {
        "WAT":None,
        "AR":None,
        "AT":None,
        "ER":None,
        "ET":None,
        "RSE":None,
        "TSE":None,
        "issame":None,
        "inp_l":None,
        "tgt_l":None,
        "out_l":None,
    }
    # return item_score
    
    output = process_string(item['output'])
    item['output_processed'] = output
    inp_l = len(item['normalized_inputs'])
    tgt_l = len(item['normalized_targets'])
    out_l = len(item['output_processed'])
    
    item['normalized_targets_entity'] = extract_entities(item['normalized_targets'])
    item['normalized_inputs_entity'] = extract_entities(item['normalized_inputs'])
    item['output_entity'] = extract_entities(item['output_processed'])
    item['evidence_entity'] = [
        extract_entities(i)
        for i in item['evidence'].split("<evidence")
    ]
    
    issame = 0
    if output.strip()==item['normalized_targets'].strip() or output.strip()==item['normalized_inputs'].strip():
        issame = 1 
        
    ar = score.edit_rouge_item(item)
    er = [
        score.rouge_evi(
            evi = evi,
            inp = item['normalized_inputs'],
            pre = output
        )
        for evi in item['evidence'].split('<evidence>')
    ]
    er = [get_score(i) for i in er]
    at = score.compute_update_and_overlap(
        item['normalized_inputs_entity'],
        item['normalized_targets_entity'],
        item['output_entity'],
        item
    )
    et = [
        score.compute_update_and_overlap_ei(
            evi = evi,
            inp = item['normalized_inputs_entity'],
            tgt = item['normalized_targets_entity'],
            outp = item['output_entity'],
        ) 
        for evi in item['evidence_entity']
    ]
    
   
    
    rse = score.edit_rouge_remaining_item(item)
    tse = score.compute_ratio_tse(
        item['normalized_inputs_entity'],
        item['normalized_targets_entity'],
        item['output_entity'],
    )
    
    
    ar = get_score(ar)
    rse = get_score(rse)
    avg_et_l = [i  for i in et  if i != -1]
    if avg_et_l==[]:
        avg_et_l = [random.random()]
    avg_et = sum(avg_et_l)/len(avg_et_l)
    et = [i  if i != -1 else avg_et for i in et]
    wat = at * len(item['evidence'].split("<evidence>"))
    
    item_score['WAT'] = wat
    item_score['AR'] = ar
    item_score["AT"] = at
    item_score['ER'] = er
    item_score["ET"] = et
    item_score['RSE'] = rse
    item_score["TSE"] = tse
    item_score["inp_l"] = inp_l
    item_score["out_l"] = out_l
    item_score["tgt_l"] = tgt_l
    item_score['issame'] = issame
    
    item['score'] = item_score
    return item

def calculate_mean_of_sublists(lst):
    return [sum(sublist) / len(sublist) if len(sublist) > 0 else 0 for sublist in lst]
def calculate_mean_per_value(data,key):
    all_values = []
    
    for item in data:
        all_values.extend(item['score'][key])
    mean_value = sum(all_values) / len(all_values) if all_values else 0
    
    return mean_value

def canculate_and_print(data, data_fn):
    "WAT↑ AR↑ AT↑ ER↑ ET↑   RSE↑ TSE↑"
    data_score = {
        "WAT":None,
        "AR":None,
        "AT":None,
        "ER":None,
        "ET":None,
        "RSE":None,
        "TSE":None,
        "inp_l":None,
        "tgt_l":None,
        "out_l":None,
    }
    data_score['AR'] = sum([i['score']['AR'] for i in data])/len(data)
    ls_at = [i for i in data if i['score']['AT']!=-1 and i['score']['AT']!=0]
    if ls_at!=[]:
        data_score['AT'] = sum([i['score']['AT'] for i in ls_at])/len(ls_at)
        print(f"len ls_at = {len(ls_at)}")
        data_score['WAT'] = sum([i['score']['WAT'] for i in ls_at])/sum([len(i['evidence'].split("<evidence>")) for i in ls_at])
        data_score['atl'] = len(ls_at)
    else:
        data_score['AT'] = 0
        data_score['WAT'] = 0
    data_score['ER'] = calculate_mean_per_value(data,'ER')
    data_score['ET'] = calculate_mean_per_value(data,'ET')
    data_score['RSE'] = sum([i['score']['RSE'] for i in data])/len(data)
    data_score['TSE'] = sum([i['score']['TSE'] for i in data])/len(data)
    
    data_score['inp_l'] = sum([i['score']['inp_l'] for i in data])/len(data)
    data_score['out_l'] = sum([i['score']['out_l'] for i in data])/len(data)
    data_score['tgt_l'] = sum([i['score']['tgt_l'] for i in data])/len(data)
    
    data_score['issame'] = sum([i['score']['issame'] for i in data])/len(data)
    pprint(data_score)
    data_fn = data_fn.split("/")[-1]
    res_dict[data_fn] = data_score
    

def process_data(data, data_fn):
    # for mistral 
    data_fn = data_fn.replace("Mistral-7B-Instruct-v0.3_0_chat","Mistral-7B-Instruct-v0.3-0")
    # if "baseline" in data_fn:
    #     return data
    # for mistral
    data = [
        process_item(i) 
        for i in tqdm(data)
    ]
    canculate_and_print(data, data_fn)
    return data

def process_one_data_fn(data_fn):
    tgt_dir = "./final_temp"
    tgt_fn = f'{tgt_dir}/{data_fn.split("/")[-1]}.ed.jsonl'
    
    data = utils.read_json(data_fn)
    
    new_data = process_data(data, data_fn)
    
    utils.write_json(tgt_fn, new_data)
    
    
import os

def get_absolute_paths(input_dir):
    abs_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            abs_paths.append(os.path.abspath(os.path.join(root, file)))
    return abs_paths
from tqdm import tqdm
if __name__ == "__main__":
    input_dir = "./final_result/llama3_1"
    for filename in tqdm(get_absolute_paths(input_dir)):
        process_one_data_fn(filename)


    pprint(res_dict)
    
    save_data_to_json(res_dict)
    pr_dict(res_dict)
