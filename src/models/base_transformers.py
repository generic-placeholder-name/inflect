import math
import torch
import torch.nn as nn
from torch.nn import Transformer
import pytorch_lightning as pl
from tokenizer import Tokenizer
from schedulers import InverseSquareLRWithWarmup

class BaseTransformer(pl.LightningModule):
    def __init__(self, tokenizer: Tokenizer, 
                 d_model=256, nhead=4, num_layers=4, d_ff=1024, dropout=0.3, # Transformers parameters
                 max_len=32, # This maximum must not be exceeded by word length + tag length * 2
                 lr=1e-3, warmup_steps=4000 # Scheduler parameters
                 ):
        super().__init__()
        self.save_hyperparameters(ignore='tokenizer')
        
        self.tokenizer = tokenizer
        self.lr = lr
        self.warmup_steps = warmup_steps
        
        # Get special tokens from tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.start_token_id = tokenizer.start_token_id
        self.end_token_id = tokenizer.end_token_id
        
        # Model components
        self.embedding = nn.Embedding(tokenizer.total_chars, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Positional encoding
        self.register_buffer("positional_encoding", 
                           self._generate_sinusoidal_encoding(max_len, d_model))
        
        self.transformer = Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, 
            dim_feedforward=d_ff, 
            activation="gelu",
            batch_first=True,
            dropout=dropout
        )
        
        self.fc_out = nn.Linear(d_model, tokenizer.vocab_size)
        
    def _generate_sinusoidal_encoding(self, max_len, d_model):
        position = torch.arange(max_len, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=self.device) * -(math.log(10000.0) / d_model))        
        encoding = torch.zeros(max_len, d_model, device=self.device)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)
        
    def forward(self, src_tags, src_forms, tgt_tags, tgt_forms, tgt_is_causal=False, 
                src_mask=None, tgt_mask=None):
        
        # Compute embeddings and apply dropout
        src_emb = self.embedding(src_forms)
        tgt_emb = self.embedding(tgt_forms)

        src_tags_emb = self.embedding(src_tags)
        tgt_tags_emb = self.embedding(tgt_tags)

        # Apply positional encoding
        src_emb += self.positional_encoding[:src_forms.shape[1]].unsqueeze(0)
        tgt_emb += self.positional_encoding[:tgt_forms.shape[1]].unsqueeze(0)
        
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)
        src_tags_emb = self.dropout(src_tags_emb)
        tgt_tags_emb = self.dropout(tgt_tags_emb)

        src_emb = torch.cat([src_tags_emb, tgt_tags_emb, src_emb], dim=1)

        # Compute masks
        src_tags_pad_mask = torch.zeros_like(src_tags, dtype=torch.bool)  # False for all tag positions
        tgt_tags_pad_mask = torch.zeros_like(tgt_tags, dtype=torch.bool)  # False for all tag positions
        src_forms_pad_mask = (src_forms == self.pad_token_id)  # True where padding
        src_key_padding_mask = torch.cat([
            src_tags_pad_mask,
            tgt_tags_pad_mask, 
            src_forms_pad_mask
        ], dim=1)
        tgt_key_padding_mask = (tgt_forms == self.pad_token_id)
        if tgt_is_causal and tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt_forms.shape[1])

        # Transformer forward pass
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        return self.fc_out(output)
    
    def training_step(self, batch, batch_idx):
        src_tags, src_forms, tgt_tags, tgt_forms = batch
        
        # Prepare target input (shift right)
        tgt_input = tgt_forms[:, :-1]
        tgt_output = tgt_forms[:, 1:]  # Shift left for loss calculation
        
        # Forward pass
        output = self(
            src_tags=src_tags,
            src_forms=src_forms,
            tgt_tags=tgt_tags,
            tgt_forms=tgt_input,
            tgt_is_causal=True
        )
        
        # Calculate loss (ignore padding)
        loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)(
            output.reshape(-1, output.size(-1)),
            tgt_output.reshape(-1)
        )
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        src_tags, src_forms, tgt_tags, tgt_forms = batch
        
        # Prepare target input (shift right)
        tgt_input = tgt_forms[:, :-1]
        tgt_output = tgt_forms[:, 1:]
        
        # Forward pass
        output = self(
            src_tags=src_tags,
            src_forms=src_forms,
            tgt_tags=tgt_tags,
            tgt_forms=tgt_input,
            tgt_is_causal=True
        )
        
        # Calculate validation loss
        loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)(
            output.reshape(-1, output.size(-1)),
            tgt_output.reshape(-1)
        )
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def generate(self, src_tags, src_forms, tgt_tags, max_len=32):
        self.eval()
        batch_size = src_tags.size(0)
        
        # Initialize output with start token
        outputs = torch.full((batch_size, 1), self.start_token_id, device=self.device)
        ended = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for _ in range(max_len):
            # Forward pass
            out = self(
                src_tags=src_tags,
                src_forms=src_forms,
                tgt_tags=tgt_tags,
                tgt_forms=outputs,
                tgt_is_causal=True
            )
            
            # Get next token
            next_tokens = out[:, -1, :].argmax(dim=-1, keepdim=True)
            outputs = torch.cat([outputs, next_tokens], dim=1)
            
            # Check for end tokens
            ended |= (next_tokens.squeeze() == self.end_token_id)
            if ended.all():
                break
        
        return outputs
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), betas=(0.99, 0.98))
        scheduler = InverseSquareLRWithWarmup(optimizer, self.lr // 100, self.lr, self.warmup_steps)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }