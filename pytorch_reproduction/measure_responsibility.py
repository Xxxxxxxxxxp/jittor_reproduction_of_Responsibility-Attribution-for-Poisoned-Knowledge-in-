#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import torch
from tqdm import tqdm

random.seed(1)


# In[2]:


def normalize(normalizationstyle,datas):
    """归一化方法"""
    if not torch.is_tensor(datas):
        datas=torch.tensor(datas)
    if normalizationstyle=='z_score':

        mean=datas.mean()
        std=datas.std().item()
        if std==0:
            return torch.zeros_like(datas)
        return (datas-mean)/std
    elif normalizationstyle=='min_max':
        if datas.max()==datas.min():
            return torch.zeros_like(datas)
        return (datas-datas.min())/(datas.max()-datas.min())



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

"""给模型看到content，要继续往下写question"""
def calculate_loss(model,tokenizer,context,question,device):
    """只有在 context + response 这个完整序列里，LLM 才能算出来"""
    """这个函数需要一个自回归的 Transformer 语言模型（如 BERT-like 的 decoder 或 GPT 类模型）来辅助计算 在给定上下文条件下生成目标文本的对数概率"""
    """从而评估文本对错误生成事件的因果支持程度"""
    text=context+' '+question
    inputs=tokenizer(text,return_tensors="pt")
    input_ids=inputs["input_ids"].to(device)
    context_ids=tokenizer(context,return_tensors="pt")["input_ids"]
    """保证长度"""
    label_ids=input_ids.clone()
    """mask掉context"""
    label_ids[:,:context_ids.shape[1]]=-100
    with torch.no_grad():
        outputs=model(input_ids, labels=label_ids)
    return outputs.loss.item()



"""特征分数"""
def calculate_scores(model,tokenizer,contexts,question,RAG_response,retrieve_score,device,trace_type=0):
    """                              contexts: List[str]"""

    generate_answer_score=[]
    generate_question_score=[]
    for idx,context in enumerate(contexts):
        prompt1=wrap_prompt_1(context,question)
        prompt2=wrap_prompt_2(context)
        """SC()"""
        answer_loss=calculate_loss(model,tokenizer,prompt2,RAG_response,device)
        generate_answer_score.append(answer_loss)
        """GC()"""
        question_loss=calculate_loss(model,tokenizer,prompt1,question,device)
        generate_question_score.append(question_loss)
    return generate_answer_score,generate_question_score, retrieve_score


def trace_scores(generate_answer_score,generate_question_score,retrieve_score,trace_type=0):
    normolize='z_score'
    generate_answer_score_norm=normalize(normolize,-np.array(generate_answer_score))
    generate_question_score_norm=normalize(normolize,-np.array(generate_question_score))
    retrieve_score_norm=normalize(normolize,np.array(retrieve_score))
    if trace_type == 0:
        return [(a + b + c) / 3 for a, b, c in zip(generate_answer_score_norm.tolist(), generate_question_score_norm.tolist(), retrieve_score_norm.tolist())]
    elif trace_type == 1:
        """GC()"""
        return generate_answer_score_norm.tolist()
    elif trace_type == 2:
        """SC()"""
        return generate_question_score_norm.tolist()
    elif trace_type == 3:
        """ES"""
        return retrieve_score_norm.tolist()


# In[4]:


def measure_responsibility(model,tokenizer,data,device,top_k,variant,trace_score_type):
    """data=[
    {
        'question_id': ...,
        'question': ...,
        'correct_answer': ...,
        'target_answer': ...,
        'RAG_response': ...,
        'as_judge_by_llm':
        'context_texts': [...],      所有文本
        'context_labels': [...],       # 每条文档是否是投毒文本（0/1）
        'retrieval_scores': [...],     # 检索阶段分数
        'check_results': [...]    # probe 阶段结果
        'check_answer'
    },"""

    y_true=[]
    """判断 1为投毒"""

    trace_score_dic={f"variant_{variant}": []}
    """trace_scores_dict={
      "variant_0": [
          [0.12, 1.83, -0.42, ...],  # 第 1 个问题
          [0.55, -0.31, 2.01, ...],  # 第 2 个问题
          ...
      ]
    }"""
    score_item=[]
    """每个元素是{
      'question_id': ...,
      'scope_size': ...,
      'answer_scores': [...],
      'question_scores': [...],
      'retrieval_scores': [...],
      'trace_scores_dict': {...}
    }"""
    ids=[]
    id_list=[]
    correct=[]
    incorrect=[]
    questions=[]
    full_context=[]
    """全部文本"""
    topk_context=[]
    """归因文本"""
    scope_sizes=[]
    for q_idx in tqdm(range(len(data))):
        item=data[q_idx]
        probe_results=item['check_result']


        """不断增加直到能完全覆盖"""
        """scope_size是有多少组
        假设前两组数据造成"""

        scope_size=2
        while scope_size < len(probe_results) and sum(probe_results[:scope_size]) != int(scope_size/2):
            scope_size+=2

        """放大"""
        scope_size=scope_size*top_k
        if scope_size>len(item['context_texts']):
            scope_size=len(item['context_texts'])
        scope_sizes.append(scope_size)





        id_list.append(item['question_id'])
        ids.append(item['question_id'])
        questions.append(item['question'])
        correct.append(item['correct_answer'])
        incorrect.append(item['target_answer'])
        full_context.append(item['context_texts'])


        contexts=item['context_texts'][:scope_size]
        topk_context.append(contexts)
        labels=item['context_labels'][:scope_size]
        scores=item['retrieval_scores'][:scope_size]
        question=item['question']
        RAG_response=item['RAG_response']
        y_true.append(labels)


        answer_scores, question_scores, retrieval_scores=calculate_scores(
            model, tokenizer, contexts, question, RAG_response, scores, device
        )
        iter_trace_scores_dict={}
        tmp_scores=trace_scores(answer_scores, question_scores, retrieval_scores, trace_score_type)
        trace_score_dic[f"variant_{variant}"].append(tmp_scores)
        iter_trace_scores_dict[f"variant_{variant}"]=tmp_scores
        score_item.append({
            'question_id': item['question_id'],
            'question': question,
            'target_answer': item['target_answer'],
            'RAG_response': RAG_response,
            'scope_size': scope_size,
            'answer_scores': answer_scores,
            'question_scores': question_scores,
            'retrieval_scores': retrieval_scores,
            'trace_scores_dict': iter_trace_scores_dict
        })

    return (
        y_true, trace_score_dic, score_item, id_list, 
        scope_sizes, topk_context, ids, questions, incorrect, correct, full_context
    )








