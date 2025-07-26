from Docdataset import Docdataset
import os
import sys 
import easyeditor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    AutoModelForCausalLM, GPT2Tokenizer,
    LlamaForCausalLM, LlamaTokenizer,
    AutoModel, AutoTokenizer,
    BloomForCausalLM, BloomTokenizerFast
)
import typing
import json
from tqdm import tqdm
from datetime import datetime
import gc
import random
import time
config = {
    "model_name":'llama-2-7b-hf',    
    'batch_size': 1,
    'log_filename': None,
    'edit_method':"base",
    'dataset_size':1
}
import argparse
def get_data_path():
    
    path = "./final_inputlabels1_100_sro_p.jsonl"
    return path
def parse_arguments():
    """
    解析命令行参数并返回结果。
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='指定模型名称'
                        ,default="llama-2-7b-c-hf" 
                        )
    parser.add_argument('--method', type=str, help='指定任务名称',
                        default="IKE"
                        )

    args = parser.parse_args()

    print(args.model)
    print(args.method)
    return args
    
def get_model_and_tokenizer(model_name):
    print("loading ", model_name)
    
    model_path = "" + model_name
    
    print("loading" + model_path)
        
    if 'gpt' in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map='auto', local_files_only=True)
            tok = GPT2Tokenizer.from_pretrained(
                model_path, padding_side='left')
            tok.pad_token_id = tok.eos_token_id
            
    elif 'llama-3' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map='auto', 
            local_files_only=True, 
            # torch_dtype=torch.bfloat16  # 设置为 FP16 精度
        )
        tok = AutoTokenizer.from_pretrained(
            model_path, padding_side='left')
        tok.pad_token_id = tok.eos_token_id
        
    elif 'llama' in model_name.lower():
        model = LlamaForCausalLM.from_pretrained(
            model_path, 
            device_map='auto', 
            local_files_only=True, 
            # torch_dtype=torch.bfloat16  # 设置为 FP16 精度
        )
        tok = LlamaTokenizer.from_pretrained(
            model_path, padding_side='left')
        tok.pad_token_id = tok.eos_token_id
    return model,tok       
def get_data():
    data = Docdataset(
        get_data_path(),
        size = config['dataset_size'],
        edit_method = config['edit_method']
        )
    dataloader = DataLoader(
        dataset=data, batch_size=config['batch_size'], shuffle=False)
    config['output_file'] = data.tgt_file_path
    # write_to_log(f"writting to {config['output_file']}")
    return dataloader

def create_log_file():
    # 获取当前的时间（月-日-小时-分钟）
    current_time = datetime.now().strftime("%m-%d-%H-%M")

    # 使用当前时间作为文件名创建日志文件
    log_filename = f"./log/log_{current_time}.txt"

    try:
        # 创建并打开日志文件
        with open(log_filename, 'w') as log_file:
            log_file.write(f"Log created at {current_time}\n")
        print(f"Log file '{log_filename}' created successfully.")
        # 更新全局配置中的日志文件名
        config['log_filename'] = log_filename
    except Exception as e:
        print(f"Error creating log file: {str(e)}")

def write_to_log(data):
    """
    Writes the given data to a log file specified in the configuration.
    - Formats dictionaries with indentation.
    - Prints lists with one element per line if the list is short.
    """
    log_filename = config.get('log_filename')
    if log_filename is not None:
        try:
            with open(log_filename, 'a') as log_file:
                if isinstance(data, dict):
                    # 写入字典，带缩进格式
                    import json
                    log_file.write(json.dumps(data, indent=4) + "\n")
                elif isinstance(data, list):
                    # 如果列表较短，每行打印一个元素；否则打印为一行
                    if len(data) <= 10:  # 设置阈值为 10
                        log_file.write('\n'.join(map(str, data)) + "\n")
                    else:
                        log_file.write(str(data) + "\n")
                else:
                    # 默认行为，将数据转换为字符串写入
                    log_file.write(str(data) + "\n")
        except Exception as e:
            print(f"Error writing to log file: {str(e)}")
    else:
        print("Log file not created. Cannot write data.")

def get_edited_model(model,item):
    return model



def remove_prefix(prefix, main_string):
    if isinstance(main_string, list) and isinstance(prefix, list):
        return [remove_prefix(p, s) for p, s in zip(prefix, main_string)]
    elif isinstance(main_string, list):
        return [remove_prefix(prefix, mains) for mains in main_string]
    if isinstance(main_string, dict):
        main_string = main_string['content']
    if main_string.startswith(prefix):
        main_string = main_string[len(prefix):]
    return main_string

def remove_prefix_with_len(prefix, main_string):
    if isinstance(main_string, list) and isinstance(prefix, list):
        return [remove_prefix(p, s) for p, s in zip(prefix, main_string)]
    elif isinstance(main_string, list):
        return [remove_prefix(prefix, mains) for mains in main_string]
    if isinstance(main_string, dict):
        main_string = main_string['content']
    main_string = main_string[len(prefix):]
    return main_string

import time
def output(model,tok,data,name):
    print('start output ' + name)
    print(f"edit_method = {config['edit_method']}")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_to_log('start output ' + name + " at " + current_time)
    a = time.time()
    show_example_input_flag = 0 
    all_output = []
    all_raw_output = []
    for i, entry in enumerate(tqdm(data)):
        model = get_edited_model(model,entry)
        real_inp = entry['real_inp']
        
        
        new_input = tok(
            real_inp,
            return_tensors="pt",
            padding=True
        ).to("cuda")
        
        with torch.no_grad():

            
            if "gpt2-xl" not in name:

                ret = model.generate(
                input_ids = new_input['input_ids'],
                attention_mask = new_input['attention_mask'],
                max_new_tokens = 512,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
                )
            else:
                if new_input['input_ids'].shape[1] > 1024:
                    ret = new_input['input_ids']  # 直接返回输入
                else:
                    ret = model.generate(
                        input_ids=new_input['input_ids'],
                        attention_mask=new_input['attention_mask'],
                        max_length=1024,  # 防止超出模型上下文长度
                        do_sample=True,
                        pad_token_id=tok.eos_token_id,
                    )

        
    
        toked_output = tok.batch_decode(
            ret, skip_special_tokens=True)
        re_toked_output = remove_prefix_with_len(
                        real_inp, toked_output)
    
        
        all_output.extend(re_toked_output)
        all_raw_output.extend(toked_output)
        
                
    log_dict = {}
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    b = time.time()
    print(f"执行时间: {b - a:.6f} 秒")
    write_to_log('end output ' + name + " at " + current_time)
    log_dict["memory_alloc_max"] = sum(
        [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])/1024/1024/1024
    log_dict["memory_res_max"] = sum(
        [torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())])/1024/1024/1024
    write_to_log(log_dict)
    
    
    data.dataset.add_value(all_output,"output")
    data.dataset.add_value(all_raw_output,"raw_output")
    
    data.dataset.write_data_to_tgt()
            
            
def main():
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    args = parse_arguments()
    config['model_name'] = args.model
    config['edit_method'] = args.method
    create_log_file() 
    data = get_data()
    model, tok = get_model_and_tokenizer(config['model_name'])
    write_to_log(config)
    
    
    output(model,tok,data,config['model_name'])
    data.dataset.print_result()
    
main()
