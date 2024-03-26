# -*- coding: utf-8 -*-

print('Load libs')
import http.server
import socketserver
import threading
import time

from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline, T5ForConditionalGeneration, T5Tokenizer
from collections import defaultdict
from bs4 import BeautifulSoup
from threading import Thread
import requests as rq
import random
import torch
import json
import time
import os
import re

import difflib
import logging
logging.getLogger('http.server').setLevel(logging.ERROR)


file='file.txt'

os.system('cls||clear')

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

    total_params = sum(p.numel() for p in model.parameters())
    wpe_weights = model.get_input_embeddings().weight
    wpe_weights_np = wpe_weights.detach().numpy()
    mt_size = list(wpe_weights.shape)[0]
    model = model.to(DEVICE)
    params = {'model': model_name_or_path, 'size': f'{int(total_params / 10**7) / 100}B', 'text': mt_size, 'device': DEVICE}
    return model, tokenizer, params


def model_log(params):
    model_name = f'Model: {params["model"]}'
    param_size = f'Size model: {params["size"]}'
    max_tokens = f'Maximum Tokens: {params["text"]}'
    device_info = f'Device: {params["device"]}'
    max_length = max([len(model_name), len(param_size), len(max_tokens), len(device_info)])
    padding = ''.zfill(max_length + 4).replace('0', '#')
    model_name = f'# {model_name}{"".zfill(max_length - len(model_name)).replace("0", " ")} #'
    param_size = f'# {param_size}{"".zfill(max_length - len(param_size)).replace("0", " ")} #'
    max_tokens = f'# {max_tokens}{"".zfill(max_length - len(max_tokens)).replace("0", " ")} #'
    device_info = f'# {device_info}{"".zfill(max_length - len(device_info)).replace("0", " ")} #'
    return f'{padding}\n{model_name}\n{param_size}\n{max_tokens}\n{device_info}\n{padding}'

def encode_ids(text, tokenizer):
    return tokenizer.encode(text, return_tensors="pt").to(DEVICE)

def generate_step_by_step(config, model, tokenizer,file):
    if True:
        text_input = config['text']
        input_ids = encode_ids(text_input, tokenizer)
        target = config['maxsize'] - len(input_ids[0])
        current_length = len(input_ids[0])
        for i in range(target):
            output = model.generate(input_ids,
                                    do_sample=config['do_sample'],
                                    temperature=config['temperature'],
                                    top_k=config['top_k'],
                                    top_p=config['top_p'],
                                    max_length=current_length + 6,
                                    pad_token_id=tokenizer.eos_token_id,
                                    num_return_sequences=config['num_return_sequences']
                                    )
            current_length += 4
            text_output = tokenizer.decode(output[0][:current_length])
            generated_text = text_output[len(config['text']):]
            '''
            if generated_text.count('import ') > 2:
                generated_text+='\nИзвените но бот не умеет писать код\n\n'
            if '```python' in generated_text:
                generated_text+='\nИзвените но бот не умеет писать код\n\n'
            '''
            if len(generated_text.split('\n')) != 1:
                return generated_text.split('\n')[0]
            else:
                '''
                cleaned_text, found_repeats = remove_repeated_phrases(generated_text)
                if found_repeats:
                    return cleaned_text
                else:
                '''
                file2=open(file,'w',encoding='utf-8')
                file2.write(text_input+generated_text)
                file2.close()
                
                os.system('cls||clear')
                print(str(text_input+generated_text).replace('Me: ','User: '))

            input_ids = encode_ids(text_output, tokenizer)
        try:
            return generated_text
        except:
            return ''

def botAw(text, model, tokenizer, params,file):
    config = {
        'text': text,
        'do_sample': True,
        'temperature': 0.5,
        'top_k': 20,
        'top_p': 0.9,
        'maxsize': params["text"],
        'num_return_sequences': 1,
    }
    generated_text = generate_step_by_step(config, model, tokenizer,file)
    return generated_text

print('Load GPT')
gpt_model_name = r'path_to_model'
gpt_model, gpt_tokenizer, gpt_params = load_model(gpt_model_name)
print(model_log(gpt_params))
print()




def var2():
    while True:
        input('Press Enter to process')
        f=open(file,'r',encoding='UTF-8')
        text=f.read()
        f.close()
                
        o=botAw(text, gpt_model, gpt_tokenizer, gpt_params,file)
                
        f=open(file,'w',encoding='UTF-8')
        f.write(text+o)
        f.close()

var2()