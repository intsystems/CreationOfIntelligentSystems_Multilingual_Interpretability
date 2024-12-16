from modules import *
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm.auto import tqdm
import warnings


def convert_to_Qwen2_ND(
    qwen
):
    qwen.to("cpu")
    for i in range(len(qwen.model.layers)):
        qwen.model.layers[i].mlp = Qwen2MLP_ND(qwen.model.layers[i].mlp)
        qwen.model.layers[i].self_attn = Qwen2SdpaAttention_ND(qwen.model.layers[i].self_attn)

    return qwen

def reset_impacts(
    qwen_nd
):
    for i in range(len(qwen_nd.model.layers)):
        qwen_nd.model.layers[i].mlp.up_proj.impacts = None
        qwen_nd.model.layers[i].mlp.down_proj.impacts = None

        qwen_nd.model.layers[i].self_attn.q_proj.impacts = None
        qwen_nd.model.layers[i].self_attn.k_proj.impacts = None
    return qwen_nd

def impacts_off(
    qwen_nd
):
    for i in range(len(qwen_nd.model.layers)):
        qwen_nd.model.layers[i].mlp.calculate_impacts = False
        qwen_nd.model.layers[i].self_attn.calculate_impacts = False
    return qwen_nd

def impacts_on(
    qwen_nd
):
    for i in range(len(qwen_nd.model.layers)):
        qwen_nd.model.layers[i].mlp.calculate_impacts = True
        qwen_nd.model.layers[i].self_attn.calculate_impacts = True
    return qwen_nd


def get_all_dsn(
    qwen_nd
):
    all_dsn = []
    for i in range(len(qwen_nd.model.layers)):
        all_dsn.append({
            "mlp": qwen_nd.model.layers[i].mlp.dsn,
            "attn": qwen_nd.model.layers[i].self_attn.dsn
        })

    return all_dsn


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
        texts = [list(elem["translation"].values())[0] for elem in batch]
        # labels = [elem.get(["label"], None) for elem in batch if elem.get(["label"], None) is not None]
        tokenized = self.tokenizer(texts, return_tensors="pt", **self.tokenizer_kwargs)
        # labels = torch.tensor(labels, dtype=torch.long)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        return input_ids, attention_mask
    

def detect_domain_specific_neurons_for_layer(
    layer,
    eps=1e-2,
    domain_name="eng",
    type="mlp", # either 'mlp' or 'attn'
    reset_impacts=False,
    reset_dsn=False # True to find DSNs iteratively
):  
    assert type in ['mlp', 'attn']

    if hasattr(layer, "dsn") and not reset_dsn and (layer.dsn is not None):
        layer_dsn = layer.dsn
    else:
        layer_dsn = {}
    layer_dsn.setdefault(domain_name, {})

    if type == 'mlp':
        if layer.up_proj.impacts is not None:
            if layer_dsn[domain_name].get("up_proj", None) is None:
                layer_dsn[domain_name]["up_proj"] = torch.all(torch.abs(layer.up_proj.impacts) > eps, dim=0).reshape(-1, 1)
            else:
                layer_dsn[domain_name]["up_proj"] = layer_dsn[domain_name]["up_proj"] * torch.all(torch.abs(layer.up_proj.impacts) > eps, dim=0).reshape(-1, 1)
            layer_dsn[domain_name]["down_proj"] = layer_dsn[domain_name]["up_proj"].reshape(1, -1)

            if reset_impacts:
                layer.up_proj.impacts = None
                layer.down_proj.impacts = None


    elif type == 'attn':
        if layer.q_proj.impacts is not None:
            if layer_dsn[domain_name].get("q_proj", None) is None:
                layer_dsn[domain_name]["q_proj"] = torch.all(torch.abs(layer.q_proj.impacts) > eps, dim=0).reshape(-1, 1)
            else:
                layer_dsn[domain_name]["q_proj"] = layer_dsn[domain_name]["q_proj"] * torch.all(torch.abs(layer.q_proj.impacts) > eps, dim=0).reshape(-1, 1)
            layer_dsn[domain_name]["k_proj"] = layer_dsn[domain_name]["q_proj"].reshape(1, -1)

            if reset_impacts:
                layer.q_proj.impacts = None
                layer.k_proj.impacts = None
    torch.cuda.empty_cache()
    layer.dsn = layer_dsn
    return layer


def calculate_impacts(
    model,
    dataloader,
    num_elements
):
    model.eval()
    outputs = []
    with torch.no_grad():
        dataloader_iter = iter(dataloader)
        with tqdm(total=num_elements) as pbar:
            i = 0
            while i < num_elements:
                try:
                    _batch, label = next(dataloader_iter)
                    pbar.update(len(_batch))
                    batch = {}
                    for k, v in _batch.items():
                        batch[k] = v.to(next(model.parameters()).device)
                    output = model(**batch)
                    outputs.append(output["logits"].cpu())
                    i += len(batch)
                except StopIteration:
                    break

    max_size = max(tensor.size(0) for tensor in outputs)

    padded_tensors = [
        F.pad(tensor, (0, 0, 0, max_size - tensor.size(0))) for tensor in outputs
    ]


    concatenated_tensor = torch.cat(padded_tensors, dim=1)

    return concatenated_tensor


def detect_domain_specific_neurons(
    model,
    tokenizer,
    dataloader=None,
    eps=1e-2,
    domain_name="eng", 
    reset_impacts=False,
    reset_dsn=True,
    num_elements=10000
):
    
    model.eval()

    if dataloader is not None:
        outputs = calculate_impacts(model, dataloader, num_elements)
    else:
        outputs = None

    
    for i in range(len(model.model.layers)):
        detect_domain_specific_neurons_for_layer(
            model.model.layers[i].mlp, 
            eps=eps, 
            domain_name=domain_name, 
            type='mlp', 
            reset_impacts=reset_impacts,
            reset_dsn=reset_dsn
        )
        detect_domain_specific_neurons_for_layer(
            model.model.layers[i].self_attn, 
            eps=eps, 
            domain_name=domain_name, 
            type='attn', 
            reset_impacts=reset_impacts,
            reset_dsn=reset_dsn
        )

    return model, outputs


def dsn_model_grads_to_train(model):

    for param in model.parameters():
        param.requires_grad = False

    for layer in model.model.layers:
        for param in layer.mlp.parameters():
            param.requires_grad = True
        for param in layer.self_attn.q_proj.parameters():
            param.requires_grad = True
        for param in layer.self_attn.k_proj.parameters():
            param.requires_grad = True
    return model


def dsn_model_mask_gradients(
    model,
    domain="eng"
):
    for i, layer in enumerate(model.model.layers):

        if layer.mlp.dsn is not None:
            layer.mlp.up_proj.weight._grad = layer.mlp.up_proj.weight._grad * (~layer.mlp.dsn[domain]["up_proj"])
            layer.mlp.down_proj.weight._grad = layer.mlp.down_proj.weight._grad * (~layer.mlp.dsn[domain]["down_proj"])
        else:
            warnings.warn(f"layers[{i}].mlp domain specific neurons mask is not defined!", UserWarning)

        if layer.self_attn.dsn is not None:
            layer.self_attn.q_proj.weight._grad = layer.self_attn.q_proj.weight._grad * (~layer.self_attn.dsn[domain]["q_proj"])
            layer.self_attn.k_proj.weight._grad = layer.self_attn.k_proj.weight._grad * (~layer.self_attn.dsn[domain]["k_proj"])
        else:
            warnings.warn(f"layers[{i}].self_attn domain specific neurons mask is not defined!", UserWarning)
    
    return model