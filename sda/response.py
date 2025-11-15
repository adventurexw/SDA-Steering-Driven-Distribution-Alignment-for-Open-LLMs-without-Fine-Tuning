import os
import gc
import time
import torch
import tqdm
from itertools import islice
import re  
import json

import numpy as np
import torch.nn.functional as F
from transformers import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial

from sda.utils import load_tokenizer_and_model  
from sda.save import *
from sda.align import *  
from sda.core import *



def safe_text(text):
    return str(text).replace("{", "{{").replace("}", "}}")

def reduce_newlines(text):
    # Replace two or more consecutive newlines with a single newline
    return re.sub(r'\n{2,}', '\n', text)

def truncate_response_by_marker(original_response, markers=["</think>"]):
    """
    Truncate the thinking process from the original response, keeping only the direct answer
    after the marker.

    Parameters:
        original_response: full response containing thinking process and answer
        markers: truncation markers (default uses "</think>")
    """
    # 1. Reduce redundant newlines
    original_response = reduce_newlines(original_response)
    for marker in markers:
        if marker in original_response:
            # 2. Find the first non-empty position after the marker
            start_idx = original_response.find(marker) + len(marker)
            response_after_marker = original_response[start_idx:].strip().replace("\n", "")
            return response_after_marker

    # If no marker is detected, return the original response
    return original_response
            

def load_models(model,device=None):
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    device_map = 'balanced_low_0'
    model_name_or_path = model
    tokenizer, model = load_tokenizer_and_model(model_name_or_path, device_map=device_map)
    model.eval()
    model.config.use_cache = True
    # Aligner-7b can be loaded on a single card (use the first available GPU)
    device = torch.device("cuda:0") 
    tokenizer_aligner = AutoTokenizer.from_pretrained("/home/xwdavid/aligner-7b-v1.0")
    model_aligner = AutoModelForCausalLM.from_pretrained(
        "/home/xwdavid/aligner-7b-v1.0",
        device_map={"": device},  # allocate to the specified device
        torch_dtype=torch.float16  # can be switched to bfloat16 depending on model requirements
    )
    # Set the aligner model to evaluation mode
    model_aligner.eval()
    return model, tokenizer, model_aligner, tokenizer_aligner

def load_base_model(model_path):
    """Load a base model (e.g., llama-2-13b-chat) and configure it for evaluation"""
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Use multi-GPU balanced loading
    device_map = 'balanced'
    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model(model_path, device_map=device_map)
    
    assert tokenizer.padding_side == "left","Tokenizer must use left padding!"
    # Set model to evaluation mode and enable cache
    model.eval()
    model.config.use_cache = True
    
    return model, tokenizer

def load_aligner_model(aligner_path="/home/xwdavid/aligner-7b-v1.0"):
    """Load the aligner model (default path) and configure it for evaluation"""
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()
    # Use the first available GPU
    device = torch.device("cuda:0")
    # Load aligner tokenizer and model
    tokenizer_aligner = AutoTokenizer.from_pretrained(aligner_path)
    model_aligner = AutoModelForCausalLM.from_pretrained(
        aligner_path,
        device_map={"": device},  # allocate to the specified device
        torch_dtype=torch.float16  # use half precision
    )
    # Set model to evaluation mode
    model_aligner.eval()
    return model_aligner, tokenizer_aligner


def load_dataset(path, start_line=0, end_line=None):
    # Load a JSONL dataset from path, returning lines in [start_line, end_line)
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in islice(f, start_line, end_line):
            data.append(json.loads(line.strip()))
    return data


def write_context(context_core):
    if context_core=="empathy":
        context = (
                "You are a warm, empathetic assistant with expertise in psychology and emotional support. "
                "Your goal is to respond with deep compassion, emotional validation, and practical guidance. "
                "Focus on understanding the user's feelings, affirm their experience, and provide helpful, non-judgmental support. "
                "Use gentle, caring language to foster trust and emotional connection in every reply.\n"
            )
    elif context_core=="harmlessness":
        context = (
                "You are a specialist in minimizing toxicity, bias, and harmful content in large language models. "
                "Your focus is on ensuring that the responses provided are safe, respectful, and free from harmful language, "
                "discrimination, or any form of offensive content. Please respond to the query in a way that is entirely harmless, "
                "avoiding any language that could be perceived as aggressive, biased, or hurtful. \n"
            )
    elif context_core=="helpfulness":
        context = (
            "You are a highly capable and helpful assistant focused on maximizing usefulness and informativeness. "
            "Your goal is to provide practical, specific, and well-structured answers that directly solve the user's problem. "
            "Prioritize clarity, completeness, usefulness, and actionable insights. "
            "Avoid vague or overly cautious language, and focus on delivering value through relevant content, examples, and step-by-step guidance.\n"
        )
    elif context_core=="reasoning":
        context = (
            "You are a capable assistant in the domain of dialogue summarization. "
            "Your goal is to generate truthful, comprehensive and well-structured summaries that capture the key points of a dialogue. "
            "Ensure that your summaries are logically coherent, concise yet comprehensive, and directly relevant to the dialogue content. "
            "Avoid hallucination, and make sure the summary would score highly on clarity, relevance, and completeness."
        )
    elif context_core=="truthful":
        context = (
                "You are a factual accuracy assistant with deep expertise in large language models and information verification. "
                "Your primary goal is to provide responses that are strictly truthful, factually accurate, and free of speculation or hallucination. "
                "When answering the following query, rely only on verified knowledge and avoid introducing any unverified or uncertain claims. "
                "If the information is unknown, unclear, or lacks strong evidence, acknowledge the limitation directly. "
                "Ensure the response is clear, and comprehensive—providing accurate information without unnecessary elaboration. \n "
            )
    else:
        raise ValueError("Invalid contextual core. Choose from 'empathy', 'harmlessness', 'helpfulness', 'reasoning', 'truthful'.")  
    return context 

def write_nc_prompt(query,context_core):
    if context_core=="reasoning":
        prompt = f"Dialog:{query}\n\nPlease summarize the dialog in the format: ['Summary': '...', 'Topic': '...']\n "
    else:
        prompt = f"{query}\n "
    return prompt


def generate_no_context_response(model, tokenizer, prompt, rate_core, max_gen_len=4096, T0=0.6, top_p=0.95, is_chat=True, flag=0):
    """
    Generate a response without context, or call to only add a context/prompt without SDA/JS adjustments.
    """
    
    if flag == 0:  # No prompt modification output
        if is_chat:
            system_prompt = get_system_prompt(rate_core)
            in_prompt = [
                {'role': 'system', 'content': system_prompt},
                {"role": "user", "content": prompt}
            ]
        else:
            in_prompt = prompt
    else:  # Only add the prompt to the output
        in_prompt = prompt

    tokenize_chat = partial(tokenizer.apply_chat_template, tokenize=True, return_tensors='pt', padding=True, return_dict=True)
    tokenize_text = partial(tokenizer, return_tensors="pt", padding=True)

    get_tokens = tokenize_chat if is_chat else tokenize_text
    inputs = get_tokens(in_prompt, add_special_tokens=True).to(model.device)  # Move inputs to the model device
    
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            do_sample=True,
            temperature=T0,
            top_p=top_p,
            max_new_tokens=max_gen_len,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,         # Stop early when encountering eos_token
        )

    # Decode and process the reply
    input_length = len(inputs.input_ids[0])
    generated_tokens = generation_output[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    response = truncate_response_by_marker(response, markers=["</think>"])  # Truncate thinking process
    return response   


def generate_prompt_response(model, tokenizer, query, nc_prompt, context, rate_core, max_gen_len=4096, T0=0.6, is_chat=True):
    """
    Generate a response that only adds the prompt (used for ablation experiments and baselines).
    """
    put_context_first = False
    if is_chat:
        new_prompts, prompts_nc = get_context_pair_dialogs([query], [nc_prompt], [context], put_context_first, rate_core=rate_core)
    else:
        new_prompts, prompts_nc = get_context_pair_texts([query], [nc_prompt], context, put_context_first)
    # print(f"Generated new_prompts: {new_prompts[0]}")
    response = generate_no_context_response(model, tokenizer, new_prompts[0], rate_core, max_gen_len, T0, is_chat, flag=1)  # flag=1 means only add the prompt
    return response


def generate_Lambda(query, response, rate_core, lambda_function=0):
    Lambda = -1
    score = -1
    count = 0  # Number of attempts, max 3
    while (Lambda == -1 and count < 3):
        try:
            Lambda, score = get_Lambda_and_score(query, response, rate_core, lambda_function)
        except Exception as e:
            print(f"Error computing Lambda: {e}, retrying...")
            count += 1
            time.sleep(5)
    return Lambda, score
    

def generate_context_response(model, tokenizer, query, prompt, context_prompt, Lambda, rate_core, max_gen_len=4096, T0=0.6, lambda_function=0, T_function=0, is_chat=True):
    """
    Generate a response using context and steering-driven alignment.
    """
    outs_adjusted = steering_driven_alignment_generation(
        model, 
        tokenizer,
        queries=[query],
        nc_prompts=[prompt],
        contexts=[context_prompt],
        lambdas=[Lambda],
        T_function=T_function,
        put_context_first=True,
        is_chat=is_chat,
        top_p=0.95,
        temperature=T0,
        show_progress=False,
        max_gen_len=max_gen_len,
        rate_core=rate_core,
    )
    if is_chat:
        context_response = outs_adjusted["generation"][0]["content"]
    else:
        context_response = outs_adjusted["generation"][0]
    context_response = truncate_response_by_marker(context_response, markers=["</think>"])  # Truncate thinking process
    return context_response



def make_aligner_prompt(query, response, rate_core):
    """
    Create the aligner correction prompt based on the specified rate_core.
    """

    if rate_core == "empathy":
        to_correct_prompt = (f"BEGINNING OF CONVERSATION: USER: Edit the following dialog pair between a human and an AI \n"
                             f"to make the answer of AI more empathetic:\n\n HUMAN:{query}\n AI: {response}\n\nASSISTANT:")
    elif rate_core == "harmlessness":
        to_correct_prompt = (f"BEGINNING OF CONVERSATION: USER: Edit the following Question-Answer pair between a human and an AI \n"
                             f"to make the answer of AI more harmless:\n\nHUMAN:{query}\nAI: {response}\n\nASSISTANT:")
    elif rate_core == "helpfulness":
        to_correct_prompt = (f"BEGINNING OF CONVERSATION: USER: Edit the following Question-Answer pair between a human and an AI \n"
                             f"to make the answer of AI more helpful:\n\nHUMAN:{query}\nAI: {response}\n\nASSISTANT:")
    elif rate_core == "reasoning":
        to_correct_prompt = (f"BEGINNING OF CONVERSATION: USER: Edit the following Dialog-Summary pair \n"
                             f"to give a better summary of the dialog:\n\nDialog:\n {query}\n Summary:\n{response}\n\n ASSISTANT:")
    elif rate_core == "truthful":
        to_correct_prompt = (f"BEGINNING OF CONVERSATION: USER: Edit the following Question-Answer pair between a human and an AI \n"
                             f"to make the answer of AI more truthful:\n\nHUMAN:{query}\nAI: {response}\n\nASSISTANT:")
    else:
        raise ValueError("Invalid rate core. Choose from 'empathy', 'harmlessness', 'helpfulness', 'reasoning', 'truthful'.")
    return to_correct_prompt




def generate_aligner_correction(tokenizer_aligner, model_aligner, query, response, rate_core, max_new_tokens=4096):
    """
    Use the aligner model to generate corrections (compatible with multi-GPU).
    """
    if rate_core == "empathy":
        to_correct_prompt = (f"BEGINNING OF CONVERSATION: USER: Edit the following Question-Answer pair "
                             f"to make it more empathetic:\n\nQ:{query}\nA: {response}\n\nASSISTANT:")
    elif rate_core == "harmlessness":
        to_correct_prompt = (f"BEGINNING OF CONVERSATION: USER: Edit the following Question-Answer pair "
                             f"to make it more harmless:\n\nQ:{query}\nA: {response}\n\nASSISTANT:")
    elif rate_core == "helpfulness":
        to_correct_prompt = (f"BEGINNING OF CONVERSATION: USER: Edit the following Question-Answer pair "
                             f"to make it more helpful:\n\nQ:{query}\nA: {response}\n\nASSISTANT:")
    elif rate_core == "reasoning":
        to_correct_prompt = (f"BEGINNING OF CONVERSATION: USER: Edit the following Question-Answer pair "
                             f"to give a better summary of the conversation:\n\nDialog:\n {query}\n Summary:\n{response}\n\n ASSISTANT:")
    elif rate_core == "truthful":
        to_correct_prompt = (f"BEGINNING OF CONVERSATION: USER: Edit the following Question-Answer pair "
                             f"to make it more truthful:\n\nQ:{query}\nA: {response}\n\nASSISTANT:")
    else:
        raise ValueError("Invalid rate core. Choose from 'empathy', 'harmlessness', 'helpfulness', 'reasoning', 'truthful'.")

    # Encode the input
    inputs = tokenizer_aligner.encode(to_correct_prompt, return_tensors='pt')

    # Check the input tensor for invalid values (e.g., NaN or Inf)
    assert not torch.isnan(inputs).any(), "Input contains NaN values!"
    assert not torch.isinf(inputs).any(), "Input contains Inf values!"

    # Move the input to the model device
    input_ids = inputs.to(model_aligner.device)

    output_ids = model_aligner.generate(
        input_ids,
        top_p=0.95,
        top_k=10,
        temperature=0.3,
        do_sample=True,
        max_new_tokens=max_new_tokens
    )[0]

    correct_aligner = tokenizer_aligner.decode(output_ids, skip_special_tokens=True)
    return find_answer_for_aligner(correct_aligner)


def save_responses_to_jsonl(data, file_path):
    """Save data to JSONL format (one JSON object per line)."""
    with open(file_path, 'a', encoding='utf-8') as f:
        # Iterate over each data item
        for item in data:
            # Write each JSON object on its own line
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()  # Ensure all data is written to disk


def save_list_to_jsonl(data_list, file_path):

    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')



def batch_generate_context_response(model, tokenizer, queries, prompts, context_prompts, Lambdas, rate_core, batch_size, max_gen_len=4096, T0=0.6, lambda_function=0, T_function=0, is_chat=True):
    outs_adjusted = steering_driven_alignment_generation(
        model, 
        tokenizer,
        queries=queries,
        nc_prompts=prompts,
        contexts=context_prompts,
        lambdas=Lambdas,
        T_function=T_function,
        put_context_first=True,
        is_chat=is_chat,
        top_p=0.95,
        temperature=T0,
        show_progress=False,
        max_gen_len=max_gen_len,
        max_batch_size=batch_size,
        rate_core=rate_core,
    )
    context_responses = []
    if is_chat:
        for out in outs_adjusted["generation"]:
            if isinstance(out, dict) and "content" in out:
                context_response = truncate_response_by_marker(out["content"], markers=["</think>"])
            else:
                context_response = ""
            context_responses.append(context_response.strip())

    else:
        for out in outs_adjusted["generation"]:
            if isinstance(out, str):
                context_response = truncate_response_by_marker(out, markers=["</think>"])
            else:
                context_response = ""
            context_responses.append(context_response.strip())
    return context_responses