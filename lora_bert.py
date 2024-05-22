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

            # Initialize lora_A with Gaussian distributed weights
            nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)  # You can adjust mean and std as needed
            
            # Initialize lora_B with zeros
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None
    
    def svd_init(self, weights):
        if self.r > 0:
            U, S, V = torch.svd(weights)
            U_r = U[:, :self.r]  # (out_features, r)
            S_r = torch.diag(S[:self.r])  # (r, r)
            V_r = V[:, :self.r].t()  # (r, in_features)
            
            self.lora_A.weight.data = V_r
            self.lora_B.weight.data = U_r @ S_r 

    def forward(self, x):
        if self.r > 0:
            return self.lora_B(self.lora_A(x))
        else:
            return 0

class LoRABertSelfAttention(BertSelfAttention):
    def __init__(self, config, r):
        super().__init__(config)

        self.query_lora = LoRALayer(config.hidden_size, self.all_head_size, r)
        self.value_lora = LoRALayer(config.hidden_size, self.all_head_size, r)

        # self.query_lora.svd_init(self.query.weight.data)
        # self.value_lora.svd_init(self.value.weight.data)

    def transform(self, x, linear_layer, lora_layer):
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x) + lora_layer(x)
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
        proj = proj.transpose(1, 2)
        return proj

    def forward(self, hidden_states, attention_mask):
        key_layer = self.transform(hidden_states, self.key, lambda x: 0)
        value_layer = self.transform(hidden_states, self.value, self.value_lora)
        query_layer = self.transform(hidden_states, self.query, self.query_lora)
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value