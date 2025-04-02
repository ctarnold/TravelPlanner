# https://medium.com/towards-agi/how-to-load-local-models-in-langchain-for-your-projects-596e3dff32be
# https://medium.com/@decodingchris/how-to-use-langchain-with-huggingface-e2fd6c971b2b
from langchain_community.llms import HuggingFacePipeline
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria

class StopWordsCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids: list):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class LocalModel:
    # consider using from HF: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    # cluster directory: /scratch/gpfs/ca2992/models/DeepSeek-R1-Distill-Llama-8B
    # cluster directory: /scratch/gpfs/ca2992/models/QwQ-32B
    # For compute clusters
    def __init__(self, model_path="/scratch/gpfs/ca2992/models/DeepSeek-R1-Distill-Llama-8B", mode = 'tool_calling'):  
        print("Loading LocalModel...", flush=True)
        self.name = model_path
        print("\nCuda Available: \n", torch.cuda.is_available())
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            # Enable flash attention if CUDA is available
            use_flash_attention = torch.cuda.is_available()
            model_kwargs = {
                "torch_dtype": torch.float16,  
                "trust_remote_code": True,
                "device_map": "auto",  
            }
            if use_flash_attention:
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    print("Using flash_attention_2")
                except Exception as e:
                    print(f"Failed to enable flash attention 2: {e}")
                    use_flash_attention = False 

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
          
            self.mode = mode
            self.model.eval() # sets model to do inference
            
            self.large_pipe = transformers.pipeline(
                    "text-generation",
                    torch_dtype=torch.float16,
                    model=self.model,
                    tokenizer= self.tokenizer,
                    device_map="auto",
                    max_new_tokens = 768,
                    do_sample=True,
                    return_full_text=False,
                    top_k=10, 
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            print("\nLarge Pipeline Initialized\n", flush=True)
            self.eval_pipe = transformers.pipeline(
                    "text-generation",
                    torch_dtype=torch.float16,
                    model=self.model,
                    tokenizer= self.tokenizer,
                    device_map="auto",
                    max_new_tokens = 1024,
                    do_sample=True,
                    return_full_text=False,
                    top_k=10, # sample for some less likely tokens when in the manager step.
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            self.tool_pipe = transformers.pipeline(
                    "text-generation",
                    torch_dtype=torch.float16,
                    model=self.model,
                    tokenizer= self.tokenizer,
                    device_map="auto",
                    max_new_tokens = 32,
                    do_sample=True,
                    return_full_text=False,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            print("\nAll Transformers Pipelines Initialized\n", flush=True)
            self.large_hf = HuggingFacePipeline(pipeline=self.large_pipe, model_kwargs={'temperature':0.1})
            self.tool_hf = HuggingFacePipeline(pipeline=self.tool_pipe, model_kwargs={'temperature':0.1})
            self.eval_pipe = HuggingFacePipeline(pipeline = self.eval_pipe, model_kwargs={'temperature': 0.1})
            print("\nHF Pipelines Initialized\n", flush=True)
            # https://stackoverflow.com/questions/76772509/llama-2-7b-hf-repeats-context-of-question-directly-from-input-prompt-cuts-off-w
            
            print("LocalModel loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise  


    def __call__(self, prompt, max_length=256, stop_list = []):
        try:
            if self.mode == 'tool_calling':
                print("\ntool hf reached\n", flush=True)
                # stop_list = ['\n']
                # stop_list.append('Action')
                # stop_list.append('Thought')
                response = self.tool_hf.invoke(prompt, stop=stop_list)
            elif self.mode == 'eval':
                print("\neval hf reached\n", flush=True)
                # stop_list = ['\n']
                response = self.eval_pipe.invoke(prompt, stop=stop_list)
            else:
                print("\nlarge hf reached\n", flush=True)
                # stop_list = ['\n']
                response = self.large_hf.invoke(prompt, stop=stop_list)
            return response
        except Exception as e:
            print(f"Error generating output: {e}", flush=True)
            return
    
    def setMode(self, mode: str):
        print("\nMode set to ", mode, "\n", flush = True)
        self.mode = mode