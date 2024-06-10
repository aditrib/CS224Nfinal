# DoRA code loosely inspired by https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/dora.py and https://github.com/catid/dora/blob/main/dora.py and 
import torch.nn as nn
import torch.nn.functional as F
from utils import *
        
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r, dora = False, freeze = False):
        super(LoRALinear, self).__init__()
        self.original_linear = original_linear  # Keep the original layer
        self.dora = dora
        self.magnitude = None

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze the original parameters, they won't be updated during training
        for param in self.original_linear.parameters():
            param.requires_grad = False

        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_B.weight)

            if dora:
                col_norms = self.original_linear.weight.norm(p=2, dim=0, keepdim=True)
                self.magnitude = nn.Parameter(col_norms)
            if freeze:
                for param in self.lora_A.parameters():
                    param.requires_grad = False
                for param in self.lora_B.parameters():
                    param.requires_grad = False
                if self.magnitude is not None:
                    self.magnitude.requires_grad = False
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        original_output = self.original_linear(x)
        if self.lora_A is not None and self.lora_B is not None:
            if self.dora:
                lora_weights = self.lora_B.weight @ self.lora_A.weight
                updated_weights = self.original_linear.weight + lora_weights
                col_norms = updated_weights.norm(p=2, dim=0, keepdim=True)
                direction = updated_weights / col_norms     
                scaled_weights = self.magnitude * direction   
                return F.linear(x, scaled_weights, self.original_linear.bias)
            else:
                lora_update = self.lora_B(self.lora_A(x))
                return original_output + lora_update
        else:
            return original_output

    
def inject_lora(bert_model, mode, r, dora = False, freeze = False):
    for layer in bert_model.bert_layers:
        # Transformer layers
        layer.self_attention.query = LoRALinear(layer.self_attention.query, r, dora, freeze)
        layer.self_attention.key = LoRALinear(layer.self_attention.key, r, dora, freeze)
        layer.self_attention.value = LoRALinear(layer.self_attention.value, r, dora, freeze)

        if mode in ['all-lin', 'all-lin-only']:
            # Other linear layers
            layer.attention_dense = LoRALinear(layer.attention_dense, r, dora, freeze)
            layer.interm_dense = LoRALinear(layer.interm_dense, r, dora, freeze)
            layer.out_dense = LoRALinear(layer.out_dense, r, dora, freeze)
    return bert_model