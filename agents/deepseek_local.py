import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class DeepSeekLocal:
    def __init__(self, model_path="../../agents/models/Qwen2.5-0.5B-Instruct", device="cuda"):  # Replace with your model path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).to(self.device)
        self.model.eval()

    def __call__(self, prompt, max_length=30000):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=max_length)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    #Alternative predict method
    #def predict(self, prompt, max_length=256):
    #    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
    #
    #    with torch.no_grad():
    #        outputs = self.model.generate(input_ids, max_length=max_length)
    #
    #    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)