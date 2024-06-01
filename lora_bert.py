import torch.nn as nn
from bert import BertSelfAttention
from utils import *

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r):
        super(LoRALayer, self).__init__()
        self.r = r
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)

            nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        if self.r > 0:
            return self.lora_B(self.lora_A(x))
        else:
            return 0
        
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r):
        super(LoRALinear, self).__init__()
        self.original_linear = original_linear  # Keep the original layer
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        for param in self.original_linear.parameters():
            param.requires_grad = False

        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        original_output = self.original_linear(x)
        if self.lora_A is not None and self.lora_B is not None:
            lora_output = self.lora_B(self.lora_A(x))
            return original_output + lora_output
        else:
            return original_output

    
def inject_lora(bert_model, mode, r):
    for layer in bert_model.bert_layers:
        # Transformer layers
        layer.self_attention.query = LoRALinear(layer.self_attention.query, r)
        layer.self_attention.key = LoRALinear(layer.self_attention.key, r)
        layer.self_attention.value = LoRALinear(layer.self_attention.value, r)

        if mode in ['all-lin', 'all-lin-only']:
            # Other linear layers
            layer.interm_dense = LoRALinear(layer.interm_dense, r)
            layer.out_dense = LoRALinear(layer.out_dense, r)
    return bert_model