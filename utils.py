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
    

def detect_domain_specific_neurons_for_layer(
    layer,
    eps=1e-2,
    domain_name="eng",
    type="mlp", # either 'mlp' or 'attn'
    reset_impacts=False,
    reset_dsn=False # True to find DSNs iteratively
):  
    assert type in ['mlp', 'attn']

    if hasattr(layer, "dsn") and not reset_dsn:
        layer_dsn = layer.dsn
    else:
        layer_dsn = {domain_name: {}}
    

    if type == 'mlp':
        if layer.up_proj.impacts is not None:
            if layer_dsn[domain_name].get("up_proj", None) is None:
                layer_dsn[domain_name]["up_proj"] = torch.all(torch.abs(layer.up_proj.impacts) > eps, dim=0)
            else:
                layer_dsn[domain_name]["up_proj"] = layer_dsn[domain_name]["up_proj"] * torch.all(torch.abs(layer.up_proj.impacts) > eps, dim=0)
            layer_dsn[domain_name]["down_proj"] = layer_dsn[domain_name]["up_proj"]

            if reset_impacts:
                layer.up_proj.impacts = None
                layer.down_proj.impacts = None


    elif type == 'attn':
        if layer.q_proj.impacts is not None:
            if layer_dsn[domain_name].get("q_proj", None) is None:
                layer_dsn[domain_name]["q_proj"] = torch.all(torch.abs(layer.q_proj.impacts) > eps, dim=0)
            else:
                layer_dsn[domain_name]["q_proj"] = layer_dsn[domain_name]["q_proj"] * torch.all(torch.abs(layer.q_proj.impacts) > eps, dim=0)
            layer_dsn[domain_name]["k_proj"] = layer_dsn[domain_name]["q_proj"]

            if reset_impacts:
                layer.q_proj.impacts = None
                layer.k_proj.impacts = None

    torch.cuda.empty_cache()
    layer.dsn = layer_dsn
    return layer


def calculate_impacts(
    model,
    dataloader
):
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch[0], batch[1])
            outputs.append(output["last_hidden_state"].cpu())
    outputs = torch.cat(outputs, dim=0)
    return output


def detect_domain_specific_neurons(
    model,
    dataloader=None,
    eps=1e-2,
    domain_name="eng", 
    reset_impacts=False,
    reset_dsn=True
):
    model.eval()

    if dataloader is not None:
        _ = calculate_impacts(model, dataloader)

    for i in range(len(model.layers)):
        detect_domain_specific_neurons_for_layer(
            model.layers[i].mlp, 
            eps=eps, 
            domain_name=domain_name, 
            type='mlp', 
            reset_impacts=reset_impacts,
            reset_dsn=reset_dsn
        )
        detect_domain_specific_neurons_for_layer(
            model.layers[i].self_attn, 
            eps=eps, 
            domain_name=domain_name, 
            type='attn', 
            reset_impacts=reset_impacts,
            reset_dsn=reset_dsn
        )

    return model