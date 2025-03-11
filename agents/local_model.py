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
    def __init__(self, model_path="../../models/DeepSeek-R1-Distill-Qwen-1.5B ", device="cuda"):  # Replace with your model path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).to(self.device)
        self.model.eval()

    def __call__(self, prompt, max_length=30000, stop_list = ['\n']):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        stopping_criteria_list = StoppingCriteriaList()

        # encodes each of the values in the stop list.
        stop_token_ids = [self.tokenizer.encode(w)[0] for w in stop_list]
        stopping_criteria_list.append(StopWordsCriteria(stop_token_ids))

        with torch.no_grad():
            outputs = self.model.generate(input_ids, 
                                          max_length=max_length, 
                                          stopping_criteria=stopping_criteria_list)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    #Alternative predict method
    #def predict(self, prompt, max_length=256):
    #    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
    #
    #    with torch.no_grad():
    #        outputs = self.model.generate(input_ids, max_length=max_length)
    #
    #    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)