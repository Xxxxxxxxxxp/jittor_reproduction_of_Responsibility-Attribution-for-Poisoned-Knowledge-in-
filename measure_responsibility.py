#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import jittor
import jittor as jt
import jittor.nn as nn
from tqdm import tqdm
from typing import List, Dict, Any, Optional
random.seed(1)
# In[ ]:





# In[1]:


MULTIPLE_PROMPT_1='Below is a query from a user and a relevant context. \
Answer the question given the information in the context. \
\n\nContext: [context] \n\nQuery: [question] \n\nAnswer:'

MULTIPLE_PROMPT_2='Below is a query from a user and a relevant context. \
Answer the question given the information in the context. \
\n\nContext: [context] \n\nQuery:'

def wrap_prompt_1(context, question) -> str:
    return MULTIPLE_PROMPT_1.replace('[context]', context).replace('[question]', question)

def wrap_prompt_2(context) -> str:
    return MULTIPLE_PROMPT_2.replace('[context]', context)
def normalize(data, style: str='z_score'):
    data=np.array(data)


    if style == 'z_score':
        mean=np.mean(data)
        std=np.std(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std

    elif style == 'min_max':
        min_v=np.min(data)
        max_v=np.max(data)
        den=max_v - min_v
        if den == 0:
            return np.zeros_like(data)
        return (data - min_v) / den

def calculate_loss(model, tokenizer, context, response):
    # 拼接并编码
    full_ids=tokenizer.encode(context + response)
    resp_ids=tokenizer.encode(response)
    
 
    input_ids=jt.array(np.array([full_ids], dtype='int32'))

    
    with jt.no_grad():
        logits=model(input_ids)
    target_len=len(resp_ids)
    shift_logits=logits[0, -(target_len+1):-1, :] 
    labels=jt.array(np.array(resp_ids, dtype='int32'))
    
   
    loss=nn.cross_entropy_loss(shift_logits, labels)
    return loss.item()

def calculate_scores(model, tokenizer, contexts, question, RAG_response, retrieval_score:List):
    """retrieval_scores直接传入 目的为形式上方便"""
    """该函数评价一个问题的所有文本contexts"""
    answer_score=[]
    generate_score=[]

    for idx,content in enumerate(contexts):
        """SC"""
        context_prompt_1=wrap_prompt_1(content, question)
        answer_loss=calculate_loss(model, tokenizer, context_prompt_1, RAG_response)
        answer_score.append(answer_loss)

        """GC"""
        context_prompt_2=wrap_prompt_2(content)
        question_loss=calculate_loss(model, tokenizer, context_prompt_2, question)
        generate_score.append(question_loss)
    return answer_score,generate_score,retrieval_score
def trace_scores(answer_scores, question_scores, retrieval_scores, normalize_style='z_score', trace_type=0):
    """归一化分数"""
    answer_scores_norm=normalize(-np.array(answer_scores))
    question_scores_norm=normalize(-np.array(question_scores))
    retrieval_scores_norm=normalize(np.array(retrieval_scores))
    if trace_type == 0:
        return [(a+b+c)/3 for a,b,c in zip(answer_scores_norm, question_scores_norm, retrieval_scores_norm)]
    elif trace_type == 1:
        return answer_scores_norm.tolist()
    elif trace_type == 2:
        return question_scores_norm.tolist()
    elif trace_type == 3:
        return retrieval_scores_norm.tolist()


# In[4]:


def measure_responsibility(data,model,tokenizer,variant,top_k):
    """初始化"""

    """[[0,1,0......],[0,1,1......]]二维数组记录每个问题每个文本"""
    y_true=[]
    """[[0.25,-0.55......],[-0.88,-0.54......]]每个问题的每个文本的分数"""
    dict_of_tracescore={f"variant_{variant}": []}
    """该单个问题的责任分数"""
    """[{},{}]"""
    scores_trace_result=[]

    id_list=[]
    ids=[]

    """归因范围"""
    scope_sizes=[]
    """范围内每个文本"""
    top_k_context=[]
    """所有文本，不限范围"""
    full_context=[]

    """原始问题"""
    question=[]

    """输出的错误答案"""
    incorrect_answer=[]
    """应该输出的正确答案"""
    correct_answer=[]


    for question_idx in tqdm(range(len(data))):
        item=data[question_idx]
        dict_of_this_tracescope={}

        probs=item['check_results']
        """初始对半分"""
        scope_size=2
        while scope_size < len(probs) and sum(probs[:scope_size]) != int(scope_size/2):
            scope_size+=2

        """放大"""
        scope_size=scope_size*top_k
        if scope_size>len(item['context_texts']):
            scope_size=len(item['context_texts'])
        scope_sizes.append(scope_size)
        
        """数据收集"""
        id_list.append(item['question_id'])
        ids.append(item['question_id'])
        question.append(item['question'])
        incorrect_answer.append(item['target_answer'])
        correct_answer.append(item['correct_answer'])
        full_context.append(item['context_texts'])


        context_text=item['context_texts'][:scope_size]
        top_k_context.append(context_text)
        context_label=item['context_labels'][:scope_size]
        retrieval_scores=item['retrieval_scores'][:scope_size]
        Question=item['question']
        RAG_response=item['RAG_response']
        y_true.append(context_label)

        answer_score,generate_score,retrieve_score=calculate_scores(model,tokenizer,context_text,Question,RAG_response,retrieval_scores)

        temp=trace_scores(answer_score,generate_score,retrieve_score,'z_score',variant)
        dict_of_tracescore[f"variant_{variant}"].append(temp)
        dict_of_this_tracescope[f"variant_{variant}"]=temp

        scores_trace_result.append({
            'question_id': item['question_id'],
            'question': Question,
            'target_answer': item['target_answer'],
            'RAG_response': RAG_response,
            'scope_size': scope_size,
            'answer_scores': answer_score,
            'question_scores': generate_score,
            'retrieval_scores': retrieve_score,
            'trace_scores_dict': dict_of_this_tracescope})


    return (y_true, dict_of_tracescore, scores_trace_result, id_list, 
        scope_sizes, top_k_context, ids, question, incorrect_answer, correct_answer, full_context)


# In[ ]:




