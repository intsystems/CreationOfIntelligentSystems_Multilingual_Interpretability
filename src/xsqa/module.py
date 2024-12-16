from transformers import Qwen2Model
import torch.nn as nn
import torch
class Qwen2ForMultipleChoice(nn.Module):
    def __init__(self, model, num_classes, torch_dtype=torch.float32):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.score = nn.Linear(896, 1,dtype = torch_dtype)
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        input_ids = input_ids.view(batch_size * self.num_classes, -1)
        attention_mask = attention_mask.view(batch_size * self.num_classes, -1)
        
        output = self.model(input_ids, attention_mask)
        output_vector = output['last_hidden_state']
        cls_index = torch.argmin(attention_mask, dim = -1)
        output_vector = output_vector.take_along_dim(cls_index[:, None, None], dim=1)
        output_vector = output_vector.reshape(batch_size, self.num_classes, -1)
        logits = self.score(output_vector)
        logits = logits.reshape(batch_size, self.num_classes)
        return {'logits': logits}
    
class Qwen2ForXLangMultipleChoice(Qwen2ForMultipleChoice):
    def __init__(self, model, num_classes, torch_dtype=torch.float32, domain_list: list[str] = ['eng']):
        super().__init__(model, num_classes, torch_dtype)
        self.embedings_for_domains: dict[str, nn.Embedding] = {}
        self.domain_list = domain_list 
        self.last_used_domain = ''
        for domain in domain_list:
            self.embedings_for_domains[domain] = self.model.embed_tokens
            self.last_used_domain = domain
    def set_domain(self, domain):
        if(domain != self.last_used_domain):
            self.embedings_for_domains[self.last_used_domain] = self.model.embed_tokens
            self.model.embed_tokens = self.embedings_for_domains[domain]