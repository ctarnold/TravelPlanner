import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    def __init__(self, model_path="/scratch/gpfs/ca2992/models/DeepSeek-R1-Distill-Llama-8B"):  
        print("Loading LocalModel...")
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

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            self.model.eval() # sets model to do inference
            print("LocalModel loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise  


    def __call__(self, prompt, max_length=30000, stop_list = ['\n']):
        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            # input_ids = self.tokenizer(prompt)
            stopping_criteria_list = StoppingCriteriaList()

            # encodes each of the values in the stop list.
            stop_token_ids = [self.tokenizer.encode(w)[0] for w in stop_list]
            stopping_criteria_list.append(StopWordsCriteria(stop_token_ids))

            with torch.no_grad():
                outputs = self.model.generate(input_ids, 
                                            max_length=max_length, 
                                            # pad_token_id=self.tokenizer.pad_token_id,
                                            stopping_criteria=stopping_criteria_list
            )
        except Exception as e:
            print(f"Error generating output: {e}")
            return

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    #Alternative predict method
    #def predict(self, prompt, max_length=256):
    #    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
    #
    #    with torch.no_grad():
    #        outputs = self.model.generate(input_ids, max_length=max_length)
    #
    #    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)