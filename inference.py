import datasets
import pandas as pd
import os
import torch
import argparse
from omegaconf import OmegaConf

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class InferenceEngine:
    def __init__(self, args):
        self.args = OmegaConf.load(args.config_path)
        self.data_config = self.args.data
        self.prompt_config = self.args.prompt
        self.model_config = self.args.model
        self.generate_config = self.args.generate
        self.vllm_generate_config = SamplingParams(**self.generate_config.vllm_gen_config)

        self.tokenizer = None
        self.model = None
        self.dt = None

    def _load_data(self, data_path):
        if "xlsx" in data_path:
            tmp_data = pd.read_excel(data_path)
            dt = datasets.Dataset.from_pandas(tmp_data)
        
        elif "csv" in data_path:
            tmp_data = pd.read_csv(data_path)
            dt = datasets.Dataset.from_pandas(tmp_data)
            
        elif "json" in data_path:
            dt = datasets.Dataset.from_json(tmp_data)
        
        return dt
        
    def make_chat(self, example):
        if "gemma" in self.tokenizer.name_or_path:
            text = self.tokenizer.apply_chat_template(
                [
                    {"role":"user","content":self.prompt_config.system_prompt + "\n" + self.prompt_config.input_prompt.format(**example)}
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.tokenizer.apply_chat_template(
                [
                    {"role":"system","content":self.prompt_config.system_prompt},
                    {"role":"user","content":self.prompt_config.input_prompt.format(**example)}
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return {"text":text}
    
    def _model_initialize(self):
        
        # Todo: Clozed Model initalize
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
        # os.environ["CUDA_VISIBLE_DEVICES"]= "0,1" # Use GPU devices list
        os.environ["CUDA_VISIBLE_DEVICES"]= ','.join([str(x) for x in range(self.model_config.device_count)]) # Use GPU devices list
        base_device_name = torch.cuda.get_device_properties(0).name

        if "A100" not in base_device_name:
            self.model_config.vllm_config.dtype = "half"

        # Todo: enable adapter
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        self.model = LLM(model = self.model_config.model_name,
            **self.model_config.vllm_config
            )
        
    def _data_initialize(self):
        '''
        This is to ensure that the Args has all the necessary settings to prepare the data. 
        It is not a preparation for data to inference into the LLM model.
        '''
        if self.generate_config.use_chat:
            if not self.prompt_config.system_prompt:
                raise ValueError(
                    "System prompt must be set when use_chat is True."
                )
            if not self.prompt_config.input_prompt:
                raise ValueError(
                    "Input prompt must be set when use_chat is True."
                )
            if not self.tokenizer:
                raise ValueError(
                    "Tokenizer must be initialized when Inference Local/Open source LLM. or Should activate function '_model_initialize' "
                )
        
    def _prepare_dataset(self):
        # for args error
        self._data_initialize()

        dataset_list = {}
        org_list = {}
        if type(self.data_config.data_path) == str:
            # Maybe single data
            if "." in self.data_config.data_path[1:]:
                file = self.data_config.data_path[:-1].split('/')[-1].split('.')[0]
                
                tmp = self._load_data(self.data_config.data_path) # It must be arrow type of HF.Dataset
                tmp_dt = tmp.map(self.make_chat, remove_columns=tmp.column_names)
                print(tmp_dt[0]['text'])
                
                dataset_list[file] = tmp_dt
                org_list[file] = tmp

            else:
                file_list = os.listdir(self.data_config.data_path)
                for file in file_list:

                    path = self.data_config.data_path
                    if path[-1] != "/":
                        path += "/"
                    
                    tmp = self._load_data(path + file)
                    tmp_dt = tmp.map(self.make_chat, remove_columns=tmp.column_names)
                    print(f"{file}의 data format 적용 후 데이터는 : {tmp_dt[0]['text']}")
                    
                    dataset_list[file] = tmp_dt
                    org_list[file] = tmp

            self.dt = dataset_list
            self.org_data = org_list

            del dataset_list, org_list
        else:
            # HF Dataset, list[str], dict
            raise ValueError(
                    "Still under development about HF Dataset, list[str], dict"
                )
    
    def _save_data(self, file_name, dataset):
        save_path = self.data_config.save_path
        if save_path[-1] != '/':
            save_path += '/'
        os.makedirs(save_path ,exist_ok=True)
        
        if "json" in self.data_config.save_type:
            dataset.to_json(save_path+f"{file_name}_inference.json")

        elif "csv" in self.data_config.save_type:
            dataset.to_csv(save_path+f"{file_name}_inference.csv")
            
        elif "xlsx" in self.data_config.save_type or "excel" in self.data_config.save_type:
            tmp = dataset.to_pandas()
            tmp.to_excel(save_path+f"{file_name}_inference.xlsx")

        else:
            raise ValueError(
                    "Output format not supported"
                )

    def inference(self):
        if not self.model or not self.tokenizer:
            self._model_initialize()
        
        if not self.dt:
            self._prepare_dataset()

        if type(self.dt) == dict :
            print("Inference in progress...!")
            for k in self.dt.keys():
                data = self.org_data[k]
                dataset = self.dt[k]
                # data : for org_data, dataset : for inference
                outputs = self.model.generate(dataset['text'], 
                                              self.vllm_generate_config
                                              )
                generated_text = [output.outputs[0].text for output in outputs]
                new_data = data.add_column(column=generated_text, name="inference_result")
                self._save_data(k, new_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="config 경로를 설정해주세요.")    
    args = parser.parse_args()

    engine = InferenceEngine(args)

    engine.inference()

if __name__ == "__main__":
    main()