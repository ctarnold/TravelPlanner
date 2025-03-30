import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from langchain.prompts import PromptTemplate
from agents.prompts import planner_agent_prompt, cot_planner_agent_prompt, react_planner_agent_prompt,reflect_prompt,react_reflect_planner_agent_prompt, planner_agent_sft_prompt, REFLECTION_HEADER
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import BaseLLM
from pydantic.v1 import BaseModel
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import tiktoken
import re
import openai
import time
from enum import Enum
from typing import List, Union, Literal
from agents.local_model import LocalModel
# from langchain_google_genai import ChatGoogleGenerativeAI
import argparse


# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
# GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']


def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.error.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.error.RateLimitError:
        print("RateLimitError")
        time.sleep(60)
    elif error == openai.error.APIError:
        print("APIError")
    elif error == openai.error.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)


class ReflexionStrategy(Enum):
    """
    REFLEXION: Apply reflexion to the next reasoning trace 
    """
    REFLEXION = 'reflexion'


class Planner:
    def __init__(self,
                 # args,
                 agent_prompt: PromptTemplate = planner_agent_prompt,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:

        self.agent_prompt = agent_prompt
        self.scratchpad: str = ''
        self.model_name = model_name
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        if model_name in  ['mistral-7B-32K']:
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=4096,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8301/v1", 
                     model_name="gpt-3.5-turbo")
        
        elif model_name in  ['ChatGLM3-6B-32K']:
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=4096,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8501/v1", 
                     model_name="gpt-3.5-turbo")
            
        elif model_name in ['mixtral']:
            self.max_token_length = 30000
            self.llm = ChatOpenAI(temperature=0,
                     max_tokens=4096,
                     openai_api_key="EMPTY", 
                     openai_api_base="http://localhost:8501/v1", 
                     model_name="YOUR/MODEL/PATH")
            
        elif model_name in ['gemini']:
            self.llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
        else:
            # self.llm = LocalModel(model_path = "../../agents/models/Qwen2.5-0.5B-Instruct")
            # self.llm = LocalModel(model_path = "/scratch/gpfs/ca2992/models/Llama-3.1-8B-Instruct-travelplanner-SFT")
            # self.llm = LocalModel(model_path = "/scratch/gpfs/ca2992/models/QwQ-32B")
            # self.llm = LocalModel(model_path = "/scratch/gpfs/ca2992/models/DeepSeek-R1-Distill-Llama-8B")
            # Llama-3.1-8B-Instruct-travelplanner-SFT
            # DeepSeek-R1-Distill-Qwen-14B
            try:
                # self.llm = LocalModel(model_path = "/scratch/gpfs/ca2992/models/Llama-3.3-70B-Instruct")
                self.llm = LocalModel(model_path = "/scratch/gpfs/ca2992/models/Llama-3.1-8B-Instruct-travelplanner-SFT")
                # self.llm = LocalModel(model_path = "/scratch/gpfs/ca2992/models/Llama-3.1-8B-Instruct", mode="planning")
            except:
                self.llm = LocalModel(model_path = "../../agents/models/Qwen2.5-0.5B-Instruct")
            
            self.llm.setMode('planner_tool')
            # self.max_token_length = 30000


        print(f"PlannerAgent {model_name} loaded.")

    def run(self, text, query, log_file=None) -> str:
        if log_file:
            log_file.write('\n---------------Planner\n'+self._build_agent_prompt(text, query))
        # print(self._build_agent_prompt(text, query))
        if self.model_name in ['gemini']:
            return str(self.llm.invoke(self._build_agent_prompt(text, query)).content)
        elif self.model_name == 'local':
            return str(self.llm(self._build_agent_prompt(text, query)))
        else:
            if len(self.enc.encode(self._build_agent_prompt(text, query))) > 12000:
                return 'Max Token Length Exceeded.'
            else:
                return self.llm([HumanMessage(content=self._build_agent_prompt(text, query))]).content

    def _build_agent_prompt(self, text, query) -> str:
        return self.agent_prompt.format(
            text=text,
            query=query)


def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None
        
    except:
        return None, None

def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

# if __name__ == '__main__':
    