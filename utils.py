from modules import *
from torch.utils.data import Dataset


def convert_to_Qwen2_ND(
    qwen
):
    qwen.to("cpu")
    for i in range(len(qwen.layers)):
        qwen.layers[i].mlp = Qwen2MLP_ND(qwen.layers[i].mlp)
        qwen.layers[i].self_attn = Qwen2SdpaAttention_ND(qwen.layers[i].self_attn)

    return qwen

def reset_impacts(
    qwen_nd
):
    for i in range(len(qwen_nd.layers)):
        qwen_nd.layers[i].mlp.up_proj.impacts = None
        qwen_nd.layers[i].mlp.down_proj.impacts = None

        qwen_nd.layers[i].self_attn.q_proj.impacts = None
        qwen_nd.layers[i].self_attn.k_proj.impacts = None
    return qwen_nd

def impacts_off(
    qwen_nd
):
    for i in range(len(qwen_nd.layers)):
        qwen_nd.layers[i].mlp.calculate_impacts = False
        qwen_nd.layers[i].self_attn.calculate_impacts = False
    return qwen_nd

def impacts_on(
    qwen_nd
):
    for i in range(len(qwen_nd.layers)):
        qwen_nd.layers[i].mlp.calculate_impacts = True
        qwen_nd.layers[i].self_attn.calculate_impacts = True
    return qwen_nd


class ImpactsDataset(Dataset):
    def __init__(self, data_path,):
        super().__init__()
        self.data = self._read_data(data_path)
    
    def _read_data(self, data_path):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data)
    
    def __get_item__(self, idx):
        raise NotImplementedError


class Collator:
    def __init__(
        self,
        tokenizer,
        max_length=512,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        **tokenizer_kwargs
    ):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        if self.tokenizer_kwargs.get("max_length", None) is None:
            self.tokenizer_kwargs["max_length"] = max_length
        if self.tokenizer_kwargs.get("padding", None) is None:
            self.tokenizer_kwargs["padding"] = padding
        if self.tokenizer_kwargs.get("truncation", None) is None:
            self.tokenizer_kwargs["truncation"] = truncation
        if self.tokenizer_kwargs.get("add_special_tokens", None) is None:
            self.tokenizer_kwargs["add_special_tokens"] = add_special_tokens
    
    def __call__(self, batch):
        texts = [elem["text"] for elem in batch]
        labels = [elem.get(["label"], None) for elem in batch if elem.get(["label"], None) is not None]
        tokenized = self.tokenizer(texts, return_tensors="pt", **self.tokenizer_kwargs)
        labels = torch.tensor(labels, dtype=torch.long)

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
    
        return input_ids, attention_mask, labels
    
