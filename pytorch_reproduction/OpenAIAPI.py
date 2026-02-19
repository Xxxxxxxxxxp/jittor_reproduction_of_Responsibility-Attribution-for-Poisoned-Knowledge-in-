

get_ipython().run_line_magic('load_ext', 'dotenv')
get_ipython().run_line_magic('dotenv', '')
import os
import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv




class OpenAIGeneratorError(Exception):
   
    pass

class OpenAIGenerator:
   

    def __init__(self, config: Dict[str, Any]):
       
        self._validate_config(config)
        self.model_name=config["generator_model"]
        self.list=config.get("generate_list", False)
        self.list_size=config.get("list_size", 5) if self.list else None
        self.batch_size=config.get("generator_batch_size", 10)
        self.generation_params=config.get("generation_params", {})

        self.client=self._initialize_client(config["openai_setting"])

    def _validate_config(self, config: Dict[str, Any]) -> None:
       
        if not config.get("generator_model"):
            raise OpenAIGeneratorError("generator_model is required but not provided.")

        if "openai_setting" not in config:
            raise OpenAIGeneratorError("openai_setting is required but not provided.")

    def _initialize_client(self, openai_setting: Dict[str, Any]) -> AsyncOpenAI:
       
        # 从环境变量获取API配置，优先使用.env文件中的设置
        api_key=os.getenv("CLOSEAI_API_KEY") or openai_setting.get("api_key")
        base_url=os.getenv("CLOSEAI_BASE_URL") or openai_setting.get("base_url", "https://api.openai-proxy.org/v1")

        if not api_key:
            raise OpenAIGeneratorError("CloseAI API key not found in environment variables or config")

        
        client_config={
            "api_key": api_key,
            "base_url": base_url,
            "max_retries": openai_setting.get("max_retries", 3),
            "timeout": openai_setting.get("timeout", 30.0),
        }

        return AsyncOpenAI(**client_config)

    async def get_response(self, messages: List[Dict[str, str]], **params) -> str:
       
        try:
            # 合并生成参数
            request_params={
                "model": self.model_name,
                "messages": messages,
                **self.generation_params,
                **params
            }

            response=await self.client.chat.completions.create(**request_params)

            if not response or not response.choices:
                raise OpenAIGeneratorError("Invalid response or no choices in the response")

            content=response.choices[0].message.content
            if content is None:
                raise OpenAIGeneratorError("No content in response")

            return content

        except Exception as e:
            raise OpenAIGeneratorError(f"Failed to get response from CloseAI: {str(e)}")

    async def get_batch_response(self, input_list: List[List[Dict[str, str]]], 
                               batch_size: int, **params) -> List[str]:
        
       
        if not input_list:
            return []

        all_results=[]

        for i in range(0, len(input_list), batch_size):
            batch=input_list[i:i + batch_size]

            tasks=[self.get_response(messages, **params) for messages in batch]

            try:
                batch_results=await asyncio.gather(*tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, Exception):
                        print(f"CloseAI API error in batch: {result}")
                        all_results.append("") 
                    else:
                        all_results.append(result)

            except Exception as e:
                print(f"CloseAI batch processing error: {e}")
                all_results.extend([""] * len(batch))

        return all_results

    async def generate(self, input_list: List[List[Dict[str, str]]], 
                batch_size: Optional[int]=None, **params) -> List[str]:
       
        if not input_list:
            return []

        batch_size=batch_size or self.batch_size

        generation_params=self.generation_params.copy()
        generation_params.update(params)

        if "n" not in generation_params:
            generation_params["n"]=1

        try:
            return await self.get_batch_response(input_list, batch_size, **generation_params)

        except Exception as e:
            raise OpenAIGeneratorError(f"CloseAI generation failed: {str(e)}")

