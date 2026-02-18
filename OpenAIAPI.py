#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'dotenv')
get_ipython().run_line_magic('dotenv', '')
from typing import List, Dict, Any, Optional
import os
import asyncio
import nest_asyncio
nest_asyncio.apply()
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
"""原作者使用from openai import AsyncOpenAI, AsyncAzureOpenAI
   但本人在复现中选择更熟悉的langchain框架"""


# In[2]:


class openeaigeneratorerror(Exception):
    pass 


# In[5]:


class openaigenerator:
    def __init__(self,config: Dict[str, Any],batchsize=5):
        """判断是否合法"""
        self._validate_config(config)
        """变量"""
        self.client=self._initialize_client(config["openai_setting"])

        self.model_name=config["generator_model"]
        self.list=config.get("generate_list", False)
        self.list_size=config.get("list_size", 5) if self.list else None
        self.batch_size=config.get("generator_batch_size", batchsize)
        self.generation_params=config.get("generation_params", {})

        """合法函数"""
    def _validate_config(self,config):
        if not config.get("generator_model"):
            openeaigeneratorerror("ERROR")
        """openai setting 在同目录下.env文件中配置"""
        if "openai_setting" not in config:
            raise openeaigeneratorerror("ERROR")

    def _initialize_client(self,openai_setting:Dict[str,Any]):
        pass
    async def get_response(self,messages=List[BaseMessage],**params)->str:
        """ messages: 例子[  SystemMessage(content="你是一个计算机老师"),
                         HumanMessage(content="解释 async/await")]"""
        model=ChatOpenAI(
            model_name=self.model_name,
            base_url=os.getenv('OPENAI_BASE_URL'),
            api_key=os.getenv('OPENAI_API_KEY1')
        )
        chain=model|StrOutputParser()
        try:
            response=await chain.ainvoke(messages)
            return response
        except Exception as e:
            raise openeaigeneratorerror(f"Failed to get response: {str(e)}")
    async def get_batch_response(self,messages_list:List[List[BaseMessage]],batchsize:int=None,**params)->List[str]:
        if batchsize==None:
            batchsize=self.batch_size
        model=ChatOpenAI(
            model_name=self.model_name,
            base_url=os.getenv('OPENAI_BASE_URL'),
            api_key=os.getenv('OPENAI_API_KEY1')
        )
        chain=model|StrOutputParser()

        if len(messages_list)==0:
            return []
        output_list=[]
        batch_size=batchsize or self.batch_size
        for i in range(0,len(messages_list),batch_size):
            batch=messages_list[i:i+batch_size]
            answers=[self.get_response(messages,**params) for messages in batch]
            try:
                batch_answers=await asyncio.gather(*answers, return_exceptions=True)
                for answer in batch_answers:
                    if isinstance(answer,Exception):
                        output_list.append("2")
                    else:
                        output_list.append(answer)
            except Exception as e:
                output_list.extend(["1"] * len(batch))
        return output_list
    def generate(self,messages:List[List[BaseMessage]],batchsize:Optional[int]=None,**params):
        if batchsize==None:
            batchsize=self.batch_size
        if len(messages)==0:
            return []
        batch_size=batchsize or self.batch_size
        generation_params=self.generation_params.copy()
        generation_params.update(params)
        if "n" not in generation_params:
            generation_params["n"]=1
        try:
            result =asyncio.run(self.get_batch_response(messages,batch_size,**generation_params))
            return result
        except Exception as e:
            raise openeaigeneratorerror(f"Generation failed: {str(e)}")




# In[ ]:





# In[ ]:




