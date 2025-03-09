import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer
import json
import random

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, d_ff=1024, max_len=32, device="cuda"):
        super(CharTransformer, self).__init__()
        self.device = device
        
        self.embedding = nn.Embedding(vocab_size, d_model).to(device)
        self.learnable_positional_encoding = nn.Parameter(torch.zeros(max_len, d_model).to(device))
        self.transformer = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, 
                                       num_decoder_layers=num_layers, dim_feedforward=d_ff).to(device)
        self.fc_out = nn.Linear(d_model, vocab_size).to(device)
        self.softmax = nn.LogSoftmax(dim=-1).to(device)
        
    def forward(self, src, tgt, feature_mask, src_mask=None, tgt_mask=None):
        src, tgt, feature_mask = src.to(self.device), tgt.to(self.device), feature_mask.to(self.device)
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        
        # Apply positional encoding only to non-feature tokens
        src_emb += (1 - feature_mask[:, :src.shape[1], None]) * self.learnable_positional_encoding[:src.shape[1], :]
        tgt_emb += (1 - feature_mask[:, :tgt.shape[1], None]) * self.learnable_positional_encoding[:tgt.shape[1], :]
        
        transformer_output = self.transformer(src_emb.permute(1,0,2), tgt_emb.permute(1,0,2), 
                                              src_mask=src_mask, tgt_mask=tgt_mask)
        output = self.fc_out(transformer_output.permute(1,0,2))
        return self.softmax(output)
    
    def generate(self, src, feature_mask, max_len=32):
        self.eval()
        src, feature_mask = src.to(self.device), feature_mask.to(self.device)
        tgt = torch.tensor([[1]], device=self.device)  # Start token index assumed as 1
        for _ in range(max_len):
            out = self.forward(src, tgt, feature_mask)
            next_char = out[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_char], dim=1)
            if next_char.item() == 2:  # End token index assumed as 2
                break
        return tgt

# Data preparation

def prepare_data(data, val_size=1000, train_size=1000, test_ratio=0.1):
    paradigms = list(data.items())
    random.shuffle(paradigms)
    
    val_set = paradigms[:val_size]
    train_test_set = paradigms[val_size:val_size + train_size]
    
    train_size = int((1 - test_ratio) * len(train_test_set))
    train_set = train_test_set[:train_size]
    test_set = train_test_set[train_size:]
    
    return train_set, test_set, val_set

def generate_examples(paradigm):
    lemma = paradigm[0]
    forms = paradigm[1]
    examples = []
    
    for tag, form in forms.items():
        src = f"<s> <{tag}> {lemma} </s>"
        tgt = form
        examples.append((src, tgt))
    
    return examples

# Load dataset
with open("data/processed/eng_v.json", "r") as f:
    data = json.load(f)

train_set, test_set, val_set = prepare_data(data, val_size=1000, train_size=1000, test_ratio=0.1)

train_examples = [ex for paradigm in train_set for ex in generate_examples(paradigm)]
test_examples = [ex for paradigm in test_set for ex in generate_examples(paradigm)]
val_examples = [ex for paradigm in val_set for ex in generate_examples(paradigm)]

print("Train examples:", train_examples[:5])  # Show first 5 examples
print("Test examples:", test_examples[:5])
print("Validation examples:", val_examples[:5])
