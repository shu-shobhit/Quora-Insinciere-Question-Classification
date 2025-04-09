import torch
import torch.nn as nn
from transformers import AutoModel

class BERT_MODEL(nn.Module):
    def __init__(self, model_name):
        super(BERT_MODEL, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.linear = nn.Sequential(
            nn.Linear(768, 1024), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(1024, 1)
        )

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        output = self.linear(pooled_output)
        return output
