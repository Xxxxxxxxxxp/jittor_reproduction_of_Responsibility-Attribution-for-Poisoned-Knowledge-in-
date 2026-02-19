

import argparse
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from determine_threshold import evaluate_result
from measure_responsibility import measure_responsibility
from narrow_scope import narrow_scope
from OpenAIAPI import OpenAIGenerator



def split_y_pred(y_pred,scope_size):
    """    y_pred：flatten后每个问题的预测结果       scope_size：  每个问题具体归因范围"""
    result=[]
    start=0
    end=0
    for size in scope_size:
        end=start+size
        if end>len(y_pred):
            end=len(y_pred)
        result.append(y_pred[start:end])
        start=end
        if start>len(y_pred):
            break
    return result


# In[3]:


def generate_config(model_name, batch_size=10, temperature=0.1, top_p=0.1):
    return {
        "generator_model": model_name,
        "generator_batch_size": batch_size,
        "generation_params": {
            "temperature": temperature,
            "top_p": top_p,
        },
        "openai_setting": {
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": os.environ.get("OPENAI_API_URL")
        },
    }

def set_up_path(args,base_dir=r"/mnt/d/code/MPT"):
    """          输入我自己本地的路径           """
    """拼接所有路径成一个目录"""
    result_dir=f'{base_dir}/{args.attack_retriever}_{args.attack_LLM}_{args.top_K}_{args.attack_M}_{args.test_version}'
    if not os.path.exists(result_dir): 
        os.makedirs(result_dir)

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
    feedback_file_path="/mnt/d/code/MPT/k5_m5_e5_gpt-4o-mini.json"
    feedback_scope_file_path=f'/mnt/d/code/MPT/{judge_model_name}_{proxy_model_name}_scope_k5_m5_e5_gpt-4o-mini.json'
    return result_dir, feedback_file_path, feedback_scope_file_path
def load_narrowed_data(feedback_file_path,feedback_scope_file_path,generator,judger,top_k):
    if not os.path.exists(feedback_scope_file_path):
        data=narrow_scope(feedback_file_path, feedback_scope_file_path, generator, judger, top_k)
    else:
        with open(feedback_scope_file_path, 'r', encoding='utf-8') as f:
            data=json.load(f)
    return data


def set_up_model(args):
    config_attack=generate_config(args.attack_LLM)
    config_judge=generate_config(args.judge_LLM)
    generator=OpenAIGenerator(config_attack)
    judge_llm=OpenAIGenerator(config_judge)

    model_name_or_path=f'{args.proxy_model}'

    tokenizer=AutoTokenizer.from_pretrained(model_name_or_path)
    model=AutoModelForCausalLM.from_pretrained(model_name_or_path)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
   # device=f"cuda:{args.cuda_device}"

    return generator, judge_llm, model, tokenizer, device


# In[ ]:


def main():
    """命令行参数解析器 在终端输入"""
    parser=argparse.ArgumentParser(description='my_experiment')
    """增加参数         名称为dataset  参数类型   默认值       """
    parser.add_argument('--dataset', type=str, default="NQ", help='Dataset name')
    parser.add_argument('--attack_retriever', type=str, default="e5", help='Attacked retriever model')
    parser.add_argument('--attack_LLM', type=str, default="deepseek-V3", help='Attacked LLM')
    parser.add_argument('--judge_LLM', type=str, default="deepseek-V3", help='LLM used for judging consistency')
    parser.add_argument('--attack_method', type=str, default="PRAGB", help='Attack method name')
    parser.add_argument('--attack_M', type=int, default=5, help='Number of poisoned documents')
    parser.add_argument('--top_K', type=int, default=5, help='Number of documents to retrieve')
    parser.add_argument('--trace_method', type=str, default="RAGOrigin", help='Tracing method')
    parser.add_argument('--proxy_model', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Model for responsibility measurement')
    parser.add_argument('--variant', type=int, default=0, help='Variant of trace score calculation')  
    parser.add_argument('--normalize_method', type=str, default="z_score_normalize", help='Method for normalizing scores')
    parser.add_argument('--feedback_root_dir', type=str, default="attack_feedback", help='Root directory for feedback files')
    parser.add_argument('--feedback_scope_dir', type=str, default="attack_feedback_scope", help='Directory for narrowed scope files')
    parser.add_argument('--result_root_dir', type=str, default="result", help='Root directory for results')
    parser.add_argument('--test_version', type=str, default="v1", help='Test version identifier')
    parser.add_argument('--cuda_device', type=int, default=1, help='CUDA device index')  
    """最终得到args传入以上函数"""
    args=parser.parse_args(args=[])
    """路径"""
    result_dire,feedback_dire,feedback_scope_dire=set_up_path(args)
    """模型"""
    generator, judge_llm, model, tokenizer, device=set_up_model(args)
    print('narrow')
    narrowed_data=load_narrowed_data(feedback_dire,feedback_scope_dire,generator,judge_llm,args.top_K)
    print('measure')

    (y_true, trace_scores_dict, scores_iter_result, id_list, scope_sizes, 
     topk_contexts, ids, questions, incorrects, corrects, full_contexts)=measure_responsibility(model,tokenizer,narrowed_data,device,args.top_K,args.variant,0)

    tn_list, fp_list, fn_list, tp_list, fpr, fnr, accuracy, y_pred=evaluate_result(y_true,trace_scores_dict,args.variant)
    split_result=split_y_pred(y_pred,scope_sizes)
    """result->list[[]]"""
    dynamic_threshold_result={
        'DACC': accuracy,
        'FPR': fpr, 
        'FNR': fnr,
        'id_list': id_list,
        'TN_list': [int(i) for i in tn_list], 
        'FP_list': [int(i) for i in fp_list],
        'FN_list': [int(i) for i in fn_list], 
        'TP_list': [int(i) for i in tp_list],
    }
    """ i：第几行->第几个问题 idx：每行第几个->每个问题中每个检索文本"""
    final_top_k=[]
    for idx1, i1 in enumerate(split_result):
        a=[]
        for idx2,i2 in enumerate(split_result[idx1]):
            if split_result[idx1][idx2]==False:
                a.append( full_contexts[idx1][idx2])
            if len(a)==args.top_K:
                break
        final_top_k.append(a)
    print(f"Top-k length: {len(final_top_k)}")
    content_result=[]
    for idx,i in enumerate(ids):
        content_result.append({'question_id': ids[idx],
            'question': questions[idx],
            'correct_answer': corrects[idx],
            'incorrect_answer': incorrects[idx],
            'clean_topk': final_top_k[idx],
            'contexts_labels': split_result[idx],
            "contexts": full_contexts[idx],
            })
    print(f"[Dynamic Threshold] FPR: {fpr}, FNR: {fnr}, DACC: {accuracy}")
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
        json.dump(content_result, f, indent=4)




# In[ ]:


if __name__ == '__main__':
    main() 





