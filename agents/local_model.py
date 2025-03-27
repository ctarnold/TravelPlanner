# https://medium.com/towards-agi/how-to-load-local-models-in-langchain-for-your-projects-596e3dff32be
# https://medium.com/@decodingchris/how-to-use-langchain-with-huggingface-e2fd6c971b2b
from langchain.llms import HuggingFacePipeline
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import StoppingCriteria, StoppingCriteriaList

class StopWordsCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids: list):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class LocalModel:
    # For local testing
    # def __init__(self, model_path="../../agents/models/Qwen2.5-0.5B-Instruct", device="cpu"):  # Replace with your model path
    #    print("Loading LocalModel...")
    #    self.device = device
    #    self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    #   self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).to(self.device)
    #    self.model.eval()
    #    print("LocalModel loaded.")
    # consider using from HF: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    # cluster directory: /scratch/gpfs/ca2992/models/DeepSeek-R1-Distill-Llama-8B
    # cluster directory: /scratch/gpfs/ca2992/models/QwQ-32B
    # For compute clusters
    def __init__(self, model_path="/scratch/gpfs/ca2992/models/DeepSeek-R1-Distill-Llama-8B", mode = 'tool_calling'):  
        print("Loading LocalModel...", flush=True)
        self.name = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            # Enable flash attention if CUDA is available
            use_flash_attention = torch.cuda.is_available()
            model_kwargs = {
                "torch_dtype": torch.bfloat16,  
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

            # self.model = AutoModelForCausalLM.from_pretrained(
            #    model_path,
            #    **model_kwargs
            # ).to(self.device)

            # TODO: Would sending to self.device work>
            # Prev observed issues where auto offload to CPU caused issues.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
          
            self.mode = mode
            self.model.eval() # sets model to do inference
            
            self.large_pipe = transformers.pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer= self.tokenizer,
                    device_map=self.device,
                    max_new_tokens = 512,
                    do_sample=True,
                    return_full_text=False,
                    top_k=30, # sample for some less likely tokens when in the manager step.
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            self.tool_pipe = transformers.pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer= self.tokenizer,
                    device_map=self.device,
                    max_new_tokens = 30,
                    do_sample=True,
                    return_full_text=False,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            self.large_hf = HuggingFacePipeline(pipeline=self.large_pipe, model_kwargs={'temperature':0.1})
            self.tool_hf = HuggingFacePipeline(pipeline=self.tool_pipe, model_kwargs={'temperature':0.1})

            # https://stackoverflow.com/questions/76772509/llama-2-7b-hf-repeats-context-of-question-directly-from-input-prompt-cuts-off-w
            
            # llm = HuggingFacePipeline.from_model_id(model_id=model_path, task="text-generation")
            print("LocalModel loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise  


    def __call__(self, prompt, max_length=256, stop_list = ['\n']):
        if self.mode == 'tool_calling':
            self.hugging_face_llm = self.tool_hf
            stop_list = ['\n']
            stop_list.append('Action')
            stop_list.append('Thought')
        else:
            stop_list = ['\n']
            self.hugging_face_llm = self.large_hf
        try:
           response = self.hugging_face_llm.invoke(prompt, stop=stop_list)
           
           return response
        except Exception as e:
            print(f"Error generating output: {e}", flush=True)
            return
    
    def setMode(self, mode: str):
        print("\nMode set to ", mode, "\n")
        self.mode = mode

    #Alternative predict method
    #def predict(self, prompt, max_length=256):
    #    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
    #
    #    with torch.no_grad():
    #        outputs = self.model.generate(input_ids, max_length=max_length)
    #
    #    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    # """