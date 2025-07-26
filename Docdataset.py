import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import typing
import transformers
import os
import utils
class Docdataset(Dataset):
    def __init__(self,
                 source_path = "",
                 edit_method = "",
                 size = None):
        self.datasetname = "Docdataset"
        self.source_path = source_path
        self.tgt_file_path = self.get_next_dataset_file_path()
        self.edit_method = edit_method
        print("data write to" + self.tgt_file_path)
    
    
        os.makedirs(os.path.dirname(self.tgt_file_path), exist_ok=True)
        if not os.path.exists(self.tgt_file_path):
            with open(self.tgt_file_path, 'w', encoding='utf-8') as file:
                json.dump([], file)
                
                
        try:
            self.data = utils.read_jsonl(self.source_path)
        except json.decoder.JSONDecodeError:
            print("load jsonl failed, try with load json")
            self.data = utils.read_json(self.source_path)
        self.data = self.process_data(self.data)
        
        if size is not None:
            self.data = self.data[:size]
            print(f'resize dataset to {size}')
        print("dataset init end, print info")
        
        print(f'sizeof dataset is {self.__len__()}')
        print(f'source_path = {source_path}')
        print("data write to" + self.tgt_file_path)
            
    def __len__(self):
        return len(self.data)
    
        
    def get_next_dataset_file_path(self):
        base_path = './result_data1'
        index = 1
        while True:
            file_name = f'{self.datasetname}{index}.json'
            file_path = os.path.join(base_path, file_name)
            if not os.path.exists(file_path):
                break
            index += 1
        return file_path

    def process_data(self,data):
            
        data = [
            {
                **item,
                "title":item['title'][0].replace("_"," "),
                "evidence":'<evidence>'.join(item['evidence']).replace("\n"," "),
                "r_evidence":'<r_evidence>'.join(item.get('r_evidence',"")).replace("\n"," ")
            }
            for item in data
        ]
        return data
        
    def write_data_to_tgt(self):
        print(self.tgt_file_path)
        with open(self.tgt_file_path, "w", encoding="utf-8") as json_file:
            json.dump(self.data, json_file, indent=4)
    
    def add_value(self,data,key):
        if len(self.data)!=len(data):
            print(f'error, len data != len output')
        for index in range(len(data)):
            self.data[index][key] = data[index]
            
    def print_result(self):
        print(self.data[0]['raw_output'])
            
    def __getitem__(self, index):
        def process_input(item):
            
            few_shot_item = {"inputs": "[0] In Mandaeism, the lofani, laufani, or laufania () is a type of ritual meal commemorating the dead. [1] It is etymologically related to the word ''laufa'' (\"spiritual communion\"), since lofani meals symbolize the connection of the souls of the living and the dead. [2] The meal sometimes contains sacrificed sheep or dove meat. [3] It is distinct from the ''zidqa brika'' and ''dukrana'', which are two other types of ritual meal offered for the dead. [CONTEXT] (0) Zidqa_brikha INTRODUCTION Unlike the lofani, which is a minor ritual meal does not require the presence of a priest, the zidqa brikha needs to be prepared by a priest. (1) Zidqa_brikha INTRODUCTION It is distinct from the ''lofani'' and ''dukrana'', which are two other types of ritual meal offered for the dead. (2) Zidqa_brikha See also *Sacred food as offering\n*Votive offering\n*Dukrana\n*Eucharist\n*Koliva\n*Lofani\n*Zidqa", "targets": "[0] [1] [2] The lofani is a minor ritual meal which does not require the presence of a priest. (0) (1) (2) It is distinct from the ''zidqa brikha'' and ''dukrana'', which are two other types of ritual meal offered for the dead. During Abu al-Haris, a day of remembrance commemorating the drowned people of Noah's flood (on the first day of the 6th Mandaean month Sar\u1e6dana), grains and cereals are eaten as part of a special lofani.", "normalized_inputs": "In Mandaeism, the lofani, laufani, or laufania () is a type of ritual meal commemorating the dead. It is etymologically related to the word ''laufa'' (\"spiritual communion\"), since lofani meals symbolize the connection of the souls of the living and the dead. The meal sometimes contains sacrificed sheep or dove meat. It is distinct from the ''zidqa brika'' and ''dukrana'', which are two other types of ritual meal offered for the dead.", "normalized_targets": "In Mandaeism, the lofani, laufani, or laufania () is a type of ritual meal commemorating the dead. It is etymologically related to the word ''laufa'' (\"spiritual communion\"), since lofani meals symbolize the connection of the souls of the living and the dead. The meal sometimes contains sacrificed sheep or dove meat. The lofani is a minor ritual meal which does not require the presence of a priest. It is distinct from the ''zidqa brikha'' and ''dukrana'', which are two other types of ritual meal offered for the dead. During Abu al-Haris, a day of remembrance commemorating the drowned people of Noah's flood (on the first day of the 6th Mandaean month Sar\u1e6dana), grains and cereals are eaten as part of a special lofani.", "evidence": ["Zidqa_brikha INTRODUCTION Unlike the lofani, which is a minor ritual meal does not require the presence of a priest, the zidqa brikha needs to be prepared by a priest.", "Zidqa_brikha INTRODUCTION It is distinct from the ''lofani'' and ''dukrana'', which are two other types of ritual meal offered for the dead.", "Zidqa_brikha See also *Sacred food as offering\n*Votive offering\n*Dukrana\n*Eucharist\n*Koliva\n*Lofani\n*Zidqa"], "id": [69663403], "title": ["Lofani"]}


            TASK_DEP = "Edit the given article to produce an updated version."
            TEMPLATE = """Acticle: {article}
            
Updated version: {article_start}"""
            EVIDENCE_TASK_DEP = "Given an article and potential evidence supporting updates, generate a updated version of the article incorporating the updates."
            
            EVIDENCE_TEMPLATE = """Acticle: {article}

Evidences: {evidence}

Updated version: {article_start}"""
            if self.edit_method in ['base','ft','memit']:
                temp = TEMPLATE
                few_shot = temp.format(
                    article = few_shot_item['normalized_inputs'],
                    article_start = few_shot_item['normalized_targets']
                )
                src = temp.format(
                    article = item['normalized_inputs'],
                    article_start = ""
                )
                ret = few_shot+'\n\n'+src
                return ret
            elif self.edit_method in ['FAME','ours',]:
                temp = EVIDENCE_TEMPLATE
                
                few_shot = temp.format(
                    article = few_shot_item['normalized_inputs'],
                    article_start = few_shot_item['normalized_targets'],
                    evidence = '\n'.join([i.replace('\n'," ") for i in few_shot_item['evidence']])
                )
                fame_e1 = item['evidence'].split("<evidence>")[:5]
                fame_e2 = "\n".join(fame_e1)
                src = temp.format(
                    article = item['normalized_inputs'],
                    article_start = "",
                    evidence = fame_e2
                )
                
                ret = few_shot+'\n\n\n\n'+src
                return ret
            elif self.edit_method in ['IKE']:
                few_shot_item = self.data[2]
                temp = EVIDENCE_TEMPLATE
                
                few_shot = temp.format(
                    article = few_shot_item['normalized_inputs'],
                    article_start = few_shot_item['normalized_targets'],
                    evidence = few_shot_item['r_evidence'].replace("<r_evidence>",'\n')
                    # evidence = '\n'.join(few_shot_item['evidence'])
                )
                src = temp.format(
                    article = item['normalized_inputs'],
                    # article_start = item['normalized_inputs'].split('.')[0]
                    article_start = "",
                    evidence = item['r_evidence'].replace("<r_evidence>",'\n')
                    # evidence = '\n'.join(item['evidence'])
                )
                
                ret = few_shot+'\n\n\n\n'+src
                return ret
            
            elif "chat" in self.edit_method:
                few_shot_item = self.data[2]
                
                if self.edit_method == 'chat_eren':
                    EREN_TASK_DEP = "Read evidences and update the article. If the article is unupdatable, say 'unupdatable'."
                
                    EREN_TEMPLATE = """Evidences: {evidence}

Acticle: {article}"""
                    ANS_TEMPLATE = "Updated version: {article_start}"
                    task_dep = EREN_TASK_DEP
                    temp = EREN_TEMPLATE
                    few_shot = temp.format(
                        article = few_shot_item['normalized_inputs'],
                        # evidence = '\n'.join([i.replace('\n'," ") for i in few_shot_item['r_evidence']])
                        # evidence = '\n'.join(few_shot_item['evidence'])
                        evidence = few_shot_item['r_evidence'].replace("<r_evidence>",'\n')
                        
                    )
                    few_shot_ans = ANS_TEMPLATE.format(
                        article_start = few_shot_item['normalized_targets'],
                    )
                    src = temp.format(
                        article = item['normalized_inputs'],
                        evidence = item['r_evidence'].replace("<r_evidence>",'\n')
                    )
                    
                elif self.edit_method == 'chat_ike':
                    CHAT_EVIDENCE_TASK_DEP = "Given an article and potential evidence supporting updates, generate a updated version of the article incorporating the updates."
                
                    CHAT_EVIDENCE_TEMPLATE = """Evidences: {evidence}

Acticle: {article}"""

                    CHAT_ANS_EVIDENCE_TEMPLATE = """Updated version: {article_start}"""
                    task_dep = CHAT_EVIDENCE_TASK_DEP
                    temp = CHAT_EVIDENCE_TEMPLATE
                    
                    few_shot = temp.format(
                        article = few_shot_item['normalized_inputs'],   
                        evidence = few_shot_item['r_evidence'].replace("<r_evidence>",'\n')
                    )
                    few_shot_ans = CHAT_ANS_EVIDENCE_TEMPLATE.format(
                        article_start = few_shot_item['normalized_targets']                        
                    )
                    src = temp.format(
                        article = item['normalized_inputs'],
                        evidence = item['r_evidence'].replace("<r_evidence>",'\n')
                        # evidence = '\n'.join(item['evidence'])
                    )
                elif self.edit_method == 'chat_fame':
                    CHAT_EVIDENCE_TASK_DEP = "Given an article and potential evidence supporting updates, generate a updated version of the article incorporating the updates."
                
                    CHAT_EVIDENCE_TEMPLATE = """Evidences: {evidence}

Acticle: {article}"""

                    CHAT_ANS_EVIDENCE_TEMPLATE = """Updated version: {article_start}"""
                    task_dep = CHAT_EVIDENCE_TASK_DEP
                    temp = CHAT_EVIDENCE_TEMPLATE
                    
                    few_shot = temp.format(
                        article = few_shot_item['normalized_inputs'],   
                        evidence = few_shot_item['r_evidence'].replace("<r_evidence>",'\n')
                        # evidence = '\n'.join(few_shot_item['evidence'])
                    )
                    few_shot_ans = CHAT_ANS_EVIDENCE_TEMPLATE.format(
                        article_start = few_shot_item['normalized_targets']                        
                    )
                    fame_ce1 = item['evidence'].split("<evidence>")[:5]
                    fame_ce2 = "\n".join(fame_ce1)
                    src = temp.format(
                        article = item['normalized_inputs'],
                        evidence = fame_ce2
                        # evidence = '\n'.join(item['evidence'])
                    )
                elif self.edit_method == 'chat_base':
                    
                    CHAT_TASK_DEP = "Edit the given article to produce an updated version."
                    CHAT_TEMPLATE = """Acticle: {article}"""
                    CHAT_ANS = """Updated version: {article_start}"""
                    task_dep = CHAT_TASK_DEP
                    temp = CHAT_TEMPLATE
                    few_shot = temp.format(
                        article = few_shot_item['normalized_inputs'],
                    )
                    few_shot_ans = CHAT_ANS.format(
                        article_start = few_shot_item['normalized_targets']
                    )
                    src = temp.format(
                        article = item['normalized_inputs'],
                    )
                    
                ret =  [
                    {"role" : "system", "content" : task_dep.strip()},
                    {"role" : "user", "content" : few_shot.strip()},
                    {"role" : "assistant", "content" : few_shot_ans.strip()},
                    {"role" : "user", "content" : src.strip()},
                ]
                pt = ct.PromptTemplate()
                pt.bulid_prompt_from_dict(ret)
                return pt.build_prompt()
            else:
                raise NotImplementedError
        self.data[index]['real_inp'] = process_input(self.data[index])
        return {
            **self.data[index],
            "normalized_inputs_sro":[],
            "normalized_targets_sro":[],
            "evidence_sro":[]
        }
