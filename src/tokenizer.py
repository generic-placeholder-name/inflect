import json
from collections import Counter

class Tokenizer:
    def __init__(self, data_files=None, pad_token_id=0, start_token_id=1, end_token_id=2):
        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id

        self.alphabet = set() # Alphabet (non-tag characters)
        
        self.char_to_id = {
            "<pad>": pad_token_id,
            "<BOS>": start_token_id,
            "<EOS>": end_token_id
        }
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.total_chars = len(self.char_to_id)
        if data_files is not None: 
            self.build_vocab(data_files)
    
    def build_vocab(self, data_files):
        all_chars = set()
        all_tags = set()
        
        for file in data_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for lemma, forms in data.items():
                    all_chars.update(lemma)
                    for tags, word in forms.items():
                        if isinstance(tags, list):
                            all_tags.update(tags)
                        else:
                            all_tags.add(tags)
                        all_chars.update(word)
        
        for char in sorted(all_chars):
            if char not in self.char_to_id:
                self.char_to_id[char] = self.total_chars
                self.id_to_char[self.total_chars] = char
                self.total_chars += 1
                
        for tag in sorted(all_tags):
            if tag not in self.char_to_id:
                self.char_to_id[tag] = self.total_chars
                self.id_to_char[self.total_chars] = tag
                self.total_chars += 1
        
        self.alphabet.update(all_chars) # Update alphabet
    
    def tokenize(self, sequence):
        return [self.char_to_id[char] for char in sequence if char in self.char_to_id]
    
    def detokenize(self, indices):
        return "".join([self.id_to_char[idx] for idx in indices if idx in self.id_to_char])
