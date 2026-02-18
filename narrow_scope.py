#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage, HumanMessage


# In[ ]:


# In[31]:


MULTIPLE_PROMPT='Below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. Your answer should be short and concise. \
If you cannot find the answer to the question, just say "I don\'t know". \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'
def wrap_prompt(context,question)->str:
    assert type(context)==list
    assert type(question)==str
    contexts="\n".join(context)
    input_prompt=MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', contexts)
    return input_prompt
def construct_input_list(context, question):
    return [SystemMessage(content= "You are a helpful assistant."),
            HumanMessage(content=wrap_prompt(context, question))]

def check(reference_answer, RAG_response):
    return [
       SystemMessage(content= "You are a helpful assistant."),
       HumanMessage(content=f'Given answer1 and answer2, answer1 is "{reference_answer}" and answer2 is "{RAG_response}". '
                            'If answer2 is exactly "I don\'t know", no matter what answer1 is, you must only just return "[Label: False]". '
                            'If answer1 is **the same as or consistant with** answer2, return "[Label: True]". '
                            'If answer1 is not consistant with answer2, return "[Label: False]".\n'
                            'Just give me the answer.')] 



# In[76]:


def process_item(item:Dict,generate_LLM,judge_LLM,top_k):
    """ 传入的item就是一个错误响应，对每个响应处理
        item为传入的dic 包括question_id
                           question
                           correct_answer
                           target_answer
                           RAG_response
                           as_judge_by_llm
                           context_texts //list[str]
                           context_labels//list[bool]
                           retrieval_scores->相似度分数，用于后续责任分数"""
    item['check_answers']=[]
    item['check_results']=[]
    item['check_answers'].append(item['RAG_response'])
    item['check_results'].append(1)
    for i in range(top_k,len(item['context_texts']),top_k):
        """从第top_k个开始：前top_k个检索结果被用来生成错误答案，已知项"""

        """early stop 继续往下成本上升，收益略少"""
        if item['check_results'] and item['check_results'].count(0) == item['check_results'].count(1):
            break
        batch_context=item['context_texts'][i:i+top_k]
        
        context_list=construct_input_list(batch_context,item['question'])
        
        generate_answer=generate_LLM.generate([context_list])
        
        judge_list=check(generate_answer,item['RAG_response'])
        judge_answer=judge_LLM.generate(judge_list)
        match=re.search(r'\[Label: (True|False)\]', judge_answer[0])

        a=0
        if match:
            if match.group(1)=='True':

                a=1
            else :
                a=0
        item['check_answers'].append(generate_answer[0])
        item['check_results'].append(a)
    return item


# In[77]:


def narrow_scope(input_file_path,feedback_file_path,generate_LLM,judge_LLM,top_k):
    
    
    with open(input_file_path,'r',encoding='utf-8') as f:
        input_file=json.load(f)

    with ThreadPoolExecutor(max_workers=5) as executor:
        processed_data=[]
        futures=[executor.submit(process_item,item,generate_LLM,judge_LLM,top_k) for item in input_file]
        for future in tqdm(as_completed(futures), total=len(futures), unit="item"):
            processed_data.append(future.result())

    with open(feedback_file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    return processed_data


# In[ ]:





# In[ ]:




