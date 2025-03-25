import random
import json
import torch
import pytorch_lightning as pl
from torch.utils import data
from tokenizer import Tokenizer

def align_sequences(src, tgt):
    # Scoring scheme
    match_score = 1
    mismatch_penalty = -3
    gap_penalty = -2
    
    m, n = len(src), len(tgt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize the DP table
    for i in range(1, m + 1):
        dp[i][0] = i * gap_penalty
    for j in range(1, n + 1):
        dp[0][j] = j * gap_penalty
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src[i - 1] == tgt[j - 1]:
                score = match_score
            else:
                score = mismatch_penalty
            dp[i][j] = max(
                dp[i - 1][j - 1] + score,  # Diagonal (match/mismatch)
                dp[i - 1][j] + gap_penalty,  # Gap in target
                dp[i][j - 1] + gap_penalty   # Gap in source
            )
    
    # Backtrack to find the aligned sequences
    aligned_src, aligned_tgt = [], []
    i, j = m, n
    while i > 0 and j > 0:
        if src[i - 1] == tgt[j - 1] or dp[i][j] == dp[i - 1][j - 1] + mismatch_penalty:
            aligned_src.append(src[i - 1])
            aligned_tgt.append(tgt[j - 1])
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + gap_penalty:
            aligned_src.append(src[i - 1])
            aligned_tgt.append('#')
            i -= 1
        else:
            aligned_src.append('#')
            aligned_tgt.append(tgt[j - 1])
            j -= 1
    
    # Add remaining characters
    while i > 0:
        aligned_src.append(src[i - 1])
        aligned_tgt.append('#')
        i -= 1
    while j > 0:
        aligned_src.append('#')
        aligned_tgt.append(tgt[j - 1])
        j -= 1
    
    # Reverse to get the correct order
    aligned_src.reverse()
    aligned_tgt.reverse()
    
    return aligned_src, aligned_tgt

class InflectionDataset(data.Dataset):
    def __init__(self, data, tokenizer, hallucinate_ratio=0.2, refresh_interval=NotImplemented, train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.hallucinate_ratio = hallucinate_ratio
        self.refresh_interval = refresh_interval
        self.train = train
        self.epochs_since_refresh = 0 # TODO: implement refreshing hallucinations.
        
        self.examples = self.generate_examples()
    
    def hallucinate(self, src, tgt):
        """
        Hallucinates an example from an existing one by replacing aligned trigrams with random characters.
        """
        # Align the sequences first
        aligned_src, aligned_tgt = self.align_sequences(src, tgt)
        n = len(aligned_src)
        
        # Find all valid trigrams (3 consecutive characters) that do not touch the ends
        trigram_indices = []
        for i in range(1, n - 3):
            if aligned_src[i] == aligned_tgt[i] and aligned_src[i + 1] == aligned_tgt[i + 1] and aligned_src[i + 2] == aligned_tgt[i + 2]:
                trigram_indices.append(i)
        random.shuffle(trigram_indices)

        # If alignment does not work, return 0
        if len(trigram_indices) == 0:
            return None, None
        
        new_src = src
        new_tgt = tgt
        replaced = [False] * n
        
        # Randomly hallucinate some of these trigrams
        for i in trigram_indices:
            if not replaced[i] and not replaced[i + 1] and not replaced[i + 2]:
                # Replace the trigram with random characters
                for j in range(i, i + 3):
                    new_src[j] = random.choice(self.tokenizer.alphabet)
                    new_tgt[j] = random.choice(self.tokenizer.alphabet)
                replaced[i] = True
                replaced[i + 1] = True
                replaced[i + 2] = True
        
        new_src = ''.join([char for char in new_src if char != '#'])
        new_tgt = ''.join([char for char in new_tgt if char != '#'])

        return new_src, new_tgt
    
    def generate_examples(self):
        """
        Generates examples from the data, including hallucinated examples if training.
        """
        examples = []
        for lemma, forms in self.data.items():
            for src_tag, src_form in forms.items():
                for tgt_tag, tgt_form in forms.items():
                    if src_tag == tgt_tag:
                        continue
                    examples.append((src_tag, src_form, tgt_tag, tgt_form))
                    if self.train and random.random() < self.hallucinate_ratio:
                        new_src_form, new_tgt_form = self.hallucinate(src_form, tgt_form)
                        if new_src_form is not None:
                            examples.extend((src_tag, new_src_form, tgt_tag, new_tgt_form)) 
        return examples
    
    def refresh_hallucinations(self):
        """
        Refreshes the hallucinated examples after a certain number of epochs.
        """
        self.epochs_since_refresh = 0
        self.examples = self.generate_examples()
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        src_tag, src_form, tgt_tag, tgt_form = self.examples[idx]
        
        # Tokenize the inputs separately
        src_tag_tokenized = self.tokenizer.tokenize(src_tag)
        src_form_tokenized = self.tokenizer.tokenize(src_form)
        tgt_tag_tokenized = self.tokenizer.tokenize(tgt_tag)
        tgt_form_tokenized = self.tokenizer.tokenize(tgt_form)
        
        return src_tag_tokenized, src_form_tokenized, tgt_tag_tokenized, tgt_form_tokenized

# PyTorch Lightning Data Module

class InflectionDataModule(pl.LightningDataModule):
    def __init__(self, data_files, batch_size=256, hallucinate_ratio=0.2, refresh_interval=None):
        super().__init__()
        self.data_files = data_files
        self.tokenizer = Tokenizer(data_files)
        self.batch_size = batch_size
        self.hallucinate_ratio = hallucinate_ratio
        self.refresh_interval = refresh_interval
        
        # Load data from JSON file
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        
        self.train_dataset = InflectionDataset(self.data, self.tokenizer, hallucinate_ratio, refresh_interval, train=True)
        self.val_dataset = InflectionDataset(self.data, self.tokenizer, train=False)
    
    def collate_fn(self, batch):
        """
        Collate function to handle padding and batching for separate inputs.
        """
        # Unpack the batch into separate lists
        src_tags, src_forms, tgt_tags, tgt_forms = zip(*batch)
        
        # Pad sequences to the same length
        src_tags = torch.nn.utils.rnn.pad_sequence(src_tags, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        src_forms = torch.nn.utils.rnn.pad_sequence(src_forms, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        tgt_tags = torch.nn.utils.rnn.pad_sequence(tgt_tags, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        tgt_forms = torch.nn.utils.rnn.pad_sequence(tgt_forms, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        # Return as a tuple in a fixed order
        return src_tags, src_forms, tgt_tags, tgt_forms
    
    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)
    
    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)