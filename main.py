#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import json
import os
import re
import jittor as jt


from determine_thredhold import evaluate_results
from measure_responsibility import measure_responsibility
from narrow_scope import narrow_scope
from OpenAIAPI import openaigenerator
from judge_LLM_training import initialize_jittor_model


# In[2]:





# In[3]:


def split_y_pred(y_pred, scope_sizes):
    result=[]
    start_index=0

    for size in scope_sizes:

        result.append(y_pred[size][:size])


    return result


# In[4]:


def split_y_pred(y_pred, scope_sizes):
    """y_pred为未压缩过的[[int],[]],scope_sizes为归因范围[int]"""

    result=[]
    start_index=0

    for size in scope_sizes:
        end_index=start_index + size
        if end_index > len(y_pred):
            end_index=len(y_pred)
        result.append(y_pred[start_index:end_index])
        start_index=end_index

        if start_index >= len(y_pred):
            break

    return result




# In[5]:


def setup_path(args):
    result_path=f'{args.result_root_dir}_{args.dataset}_{args.attack_method}_{args.trace_method}_{args.attack_retriever}_{args.attack_LLM}_{args.top_K}_{args.attack_M}_{args.test_version}_jt'
    if not os.path.exists(result_path): 
        os.makedirs(result_path)
    proxy_model_name=args.proxy_model
    if 'meta-llama/' in proxy_model_name:
        proxy_model_name=proxy_model_name.replace('meta-llama/','')
    judge_model_name=args.attack_LLM
    if 'Qwen/' in proxy_model_name:
        proxy_model_name=proxy_model_name.replace('Qwen/','')
    if 'meta-llama/' in judge_model_name:
        judge_model_name=judge_model_name.replace('meta-llama/','')
    if 'openai/' in judge_model_name:
        judge_model_name=judge_model_name.replace('openai/','')
    feedback_file_path='/mnt/d/code/MPT/k5_m5_e5_gpt-4o-mini.json'
    feedback_scope_file_path=f'/mnt/d/code/MPT/{judge_model_name}_{proxy_model_name}_scope_k5_m5_e5_gpt-4o-mini.json'
    return result_path,feedback_file_path,feedback_scope_file_path
def load_narrowed_data(feedback_file_path, feedback_scope_file_path, generator, judge_llm, top_K):
    if not os.path.exists(feedback_scope_file_path):
        data=narrow_scope(feedback_file_path, feedback_scope_file_path, generator, judge_llm, top_K)
        print('narrow_scope')

    else:
        with open(feedback_scope_file_path, 'r', encoding='utf-8') as f:
            data=json.load(f)
    return data


# In[9]:


def generate_config(model_name, batch_size=10, temperature=0.1, top_p=0.1):

    return {
        "generator_model": model_name,
        "generator_batch_size": batch_size,
        "generation_params": {
            "temperature": temperature,
            "top_p": top_p},
        "openai_setting": {
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": os.environ.get("OPENAI_API_URL")
        },
    }
def setup_model(args,judge_config):
    config=generate_config(args.attack_LLM)
    config_judge_llm=generate_config(args.judge_LLM)
    generator=openaigenerator(config)
    judge_llm=openaigenerator(config_judge_llm)

    model_name_or_path=f'{args.proxy_model}'
    model,tokenizer=initialize_jittor_model(model_name_or_path,judge_config)
    model.eval()

    return generator,judge_llm,model,tokenizer


# In[10]:


def main():
    parser=argparse.ArgumentParser(description='Experiment')
    parser.add_argument('--dataset', type=str, default="NQ", help='Dataset name')
    parser.add_argument('--attack_retriever', type=str, default="e5", help='Attacked retriever model')
    parser.add_argument('--attack_LLM', type=str, default="gpt-4o-mini", help='Attacked LLM')
    parser.add_argument('--judge_LLM', type=str, default="gpt-4o-mini", help='LLM used for judging consistency')
    parser.add_argument('--attack_method', type=str, default="PRAGB", help='Attack method name')
    parser.add_argument('--attack_M', type=int, default=5, help='Number of poisoned documents')
    parser.add_argument('--top_K', type=int, default=5, help='Number of documents to retrieve')
    parser.add_argument('--trace_method', type=str, default="RAGOrigin", help='Tracing method')
    parser.add_argument('--proxy_model', type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help='Model for responsibility measurement')
    parser.add_argument('--variant', type=int, default=0, help='Variant of trace score calculation')  
    parser.add_argument('--normalize_method', type=str, default="z_score_normalize", help='Method for normalizing scores')
    parser.add_argument('--feedback_root_dir', type=str, default="/mnt/d/code/MPT", help='Root directory for feedback files')
    parser.add_argument('--feedback_scope_dir', type=str, default="/mnt/d/code/MPT", help='Directory for narrowed scope files')
    parser.add_argument('--result_root_dir', type=str, default="/mnt/d/code/MPT/result", help='Root directory for results')
    parser.add_argument('--test_version', type=str, default="v1", help='Test version identifier')
    parser.add_argument('--cuda_device', type=int, default=1, help='CUDA device index')  
    args=parser.parse_args(args=[])

    result_path,feedback_file_path,feedback_scope_file_path=setup_path(args)

    LLaMA_3_1_3Bconfig={
        "vocab_size": 128256,
        "hidden_size": 3072,           
        "intermediate_size": 8192,     
        "num_hidden_layers": 28,      
        "num_attention_heads": 24,     
        "num_key_value_heads": 8,       
        "max_position_embeddings": 131072,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings":True,
        "attention_bias": False  
    }
    # Qwen2.5-0.5B-Instruct 配置
    Qwen_2_5_0B_config={
        "vocab_size": 151936,           
        "hidden_size": 896,           
        "intermediate_size": 4864,      
        "num_hidden_layers": 24,        
        "num_attention_heads": 14,      
        "num_key_value_heads": 2,       
        "max_position_embeddings": 32768, 
        "rope_theta": 1000000.0,        
        "rms_norm_eps": 1e-6,           
        "tie_word_embeddings": True,    
        "attention_bias": True          
    }
    # Qwen2.5-1.5B-Instruct 配置
    Qwen_2_5_1B_config={
        "vocab_size": 151936,           # Qwen 统一大词表
        "hidden_size": 1536,            # 1.5B 特有维度
        "intermediate_size": 8960,      # MLP 维度
        "num_hidden_layers": 28,        # 层数 (和 Llama 3B 一样，但维度小)
        "num_attention_heads": 12,      # 注意力头数 (1536 / 12=128 head_dim)
        "num_key_value_heads": 2,       # GQA (2组 KV, 也就是 6:1 的比例)
        "max_position_embeddings": 32768, # 32k 上下文
        "rope_theta": 1000000.0,        # Qwen theta
        "rms_norm_eps": 1e-6,           # 1e-6
        "tie_word_embeddings": True,    # 1.5B 也是共享权重的
        "attention_bias": True          # 【必须】开启 QKV 的 Bias
    }
    # Qwen2.5-3B-Instruct 官方配置
    Qwen_2_5_3B_config={
        "vocab_size": 151936,           # Qwen 统一大词表
        "hidden_size": 2560,            # 3B 特有维度
        "intermediate_size": 6912,      # MLP 维度
        "num_hidden_layers": 36,        # 层数增加到 36 层
        "num_attention_heads": 20,      # 2560 / 20=128 head_dim
        "num_key_value_heads": 4,       # GQA (5:1 比例)
        "max_position_embeddings": 32768, 
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,   # 【关键差异】3B 不共享权重，lm_head 是独立的
        "attention_bias": True          # Qwen2.5 全系开启 Bias
    }
    generator,judge_llm,model,tokenizer=setup_model(args,Qwen_2_5_1B_config)

    data=load_narrowed_data(feedback_file_path, feedback_scope_file_path, generator, judge_llm, args.top_K)

    print("数据长度:", len(data))



    (y_true, trace_scores_dict, scores_iter_result, id_list, scope_sizes, 
     topk_contexts, ids, questions, incorrects, corrects, full_contexts)=measure_responsibility(data, model, tokenizer, args.variant, args.top_K)

    tn_list,fp_list,fn_list,tp_list,fpr,fnr,accuracy_final,accuracy_list,y_pred=evaluate_results(y_true, trace_scores_dict, args.variant)

    split_result =y_pred

    dynamic_threshold_result={
        'DACC_LIST':accuracy_list,
        'DACC': accuracy_final,
        'FPR': fpr, 
        'FNR': fnr,
        'id_list': id_list,
        'TN_list': [int(i) for i in tn_list], 
        'FP_list': [int(i) for i in fp_list],
        'FN_list': [int(i) for i in fn_list], 
        'TP_list': [int(i) for i in tp_list],
    }
    topk=[]
    for idx, i in enumerate(split_result):
        tmp=[]
        for idx2, i2 in enumerate(split_result[idx]):
            if split_result[idx][idx2] == False:  # False means clean context
                tmp.append(full_contexts[idx][idx2])
            if len(tmp) == args.top_K:
                break
        topk.append(tmp)
    print(f"Top-k length: {len(topk)}")

    context_result=[]
    for idx, i in enumerate(ids):
        context_result.append({
            'question_id': ids[idx],
            'question': questions[idx],
            'correct_answer': corrects[idx],
            'incorrect_answer': incorrects[idx],
            'clean_topk': topk[idx],
            'contexts_labels': split_result[idx],
            "contexts": full_contexts[idx],
        })

    print(f"[Dynamic Threshold] FPR: {fpr}, FNR: {fnr}, DACC: {accuracy_final}")

    metric_result={
        'dynamic_threshold_result': dynamic_threshold_result,
    }
    result_dir='/mnt/d/code/MPT'
    proxy_model_name=args.proxy_model
    if 'meta-llama/' in proxy_model_name:
        proxy_model_name=proxy_model_name.replace('meta-llama/','')
    judge_model_name=args.attack_LLM
    if 'meta-llama/' in judge_model_name:
        judge_model_name=judge_model_name.replace('meta-llama/','')
    if 'Qwen/' in proxy_model_name:
        proxy_model_name=proxy_model_name.replace('Qwen/','')
    with open(f'{result_dir}/{judge_model_name}_{proxy_model_name}_scores_iter_result.json', 'w') as f:
        json.dump(scores_iter_result, f, indent=4)

    with open(f'{result_dir}/{judge_model_name}_{proxy_model_name}_metric_result.json', 'w') as f:
        json.dump(metric_result, f, indent=4)

    with open(f'{result_dir}/{judge_model_name}_{proxy_model_name}_context_result.json', 'w') as f:
        json.dump(context_result, f, indent=4)



# In[8]:


if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




