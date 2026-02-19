

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import asyncio
import nest_asyncio


#处理提示词
MULTIPLE_PROMPT='Below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. Your answer should be short and concise. \
If you cannot find the answer to the question, just say "I don\'t know". \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'
def wrap_prompt(question,contexts)->str:
    #传入问题和检索的上下文并指定return string
    assert type(contexts)==list
    splitted_context='\n'.join(contexts)
    #print(type(splitted_context))-->str
    #分离传入的列表
    prompt=MULTIPLE_PROMPT.replace(' [context] ',splitted_context).replace('[question]',question)
    return prompt
              #question必须为str
def input_prompt_list(question, contexts):
    return [{"role": "system", "content": "You are a helpful assistant your objective is to answer user's question."},
            {"role": "user", "content": wrap_prompt(question, contexts)}]   
    #最终应该输入给LLM的prompt





def checking(subset_answer,RAG_answer):
    judging_prompt_list=[ {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": (
            f'Given answer1 and answer2, answer1 is "{subset_answer}" and answer2 is "{RAG_answer}". '
            'If answer2 is exactly "I don\'t know", no matter what answer1 is, you must only just return "[Label: False]". '
            'If answer1 is **the same as or consistant with** answer2, return "[Label: True]". '
            'If answer1 is not consistant with answer2, return "[Label: False]".\n'
            'Just give me the answer.')}]
    return judging_prompt_list
    #为什么不能直接用embedder+cos（）相似度


async def process_item(item,generator_LLM,judger_LLM,top_k,sem):
    #传入的item就是一个错误响应，对每个响应处理
    # item为传入的dic 包括question_id
    #                    question
    #                    correct_answer
    #                    target_answer
    #                    RAG_response
    #                    as_judge_by_llm
    #                    context_texts //list[str]
    #                    context_labels//list[bool]
    #                    retrieval_scores->相似度分数，用于后续责任分数


    question=item['question']
    context_texts=item['context_texts']
    rag_answer=item['RAG_response']

    item['check_result']=[]
    item['check_answer']=[]
    item['narrowed_context']=None

    for i in range(0, len(context_texts), top_k):

        subset=context_texts[i:i+top_k]

        subset_prompt=input_prompt_list(question, subset)
        async with sem:
            subset_answer=await generator_LLM.get_response(subset_prompt)

        judge_prompt=checking(subset_answer, rag_answer)
        async with sem:
            judged_answer=await judger_LLM.get_response(judge_prompt)

        match=re.search(r'\[Label: (True|False)\]', judged_answer)

        if match:
            label=match.group(1)
            is_consistent=1 if label == "True" else 0
        else:
            is_consistent=0

        item['check_result'].append(is_consistent)
        item['check_answer'].append(judged_answer)

        # 找到第一个 consistent 子集
        if is_consistent == 1:
            item['narrowed_context']=subset
            break

    return item




# In[11]:


def narrow_scope(input_json, output_json, generator_LLM, judger_LLM, top_k):
    sem=asyncio.Semaphore(2)
    with open(input_json, 'r', encoding='utf-8') as f:
        data=json.load(f)
    narrow_item=[]
    async def run_processing():
        # 创建所有异步任务
        
        tasks=[process_item(item, generator_LLM, judger_LLM, top_k,sem) for item in data]
        
        narrow_results=[]
        # 使用 asyncio.as_completed 配合 tqdm 显示进度
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Items"):
            result=await f  # 真正等待异步任务执行完毕
            narrow_results.append(result)
        return narrow_results

    # 3. 同步调用异步的核心桥梁
    print("Starting asynchronous tasks...")
    try:
        # 正常脚本运行
        narrow_item=asyncio.run(run_processing())
    except RuntimeError:
        # 如果在 Jupyter Notebook 或已有 Loop 的环境下运行
        
        nest_asyncio.apply()
        loop=asyncio.get_event_loop()
        narrow_item=loop.run_until_complete(run_processing())

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(narrow_item, f, ensure_ascii=False, indent=4)

    return narrow_item


# In[ ]:




