import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


def get_num_class(s, num1, num2):
    if s == "accu":
        return num2
    elif s == "law":
        return num1
    else:
        return 11
    
def load_law_text(law_path):
    law_text = []
    with open(law_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip('\n')) > 254:
                law_text.append(line.strip('\n')[:254])
            else:
                law_text.append(line.strip('\n'))
    return law_text

def load_charge_text(charge_path):
    charge_text = []
    with open(charge_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip('\n')) > 254:
                charge_text.append(line.strip('\n')[:254])
            else:
                charge_text.append(line.strip('\n'))
    return charge_text
    
def load_data(data_path):
    fact_text = []
    law_label = []
    accu_label = []
    term_label = []

    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            samples.append(data)
    for sample in samples:
        if len(sample['fact_cut'].replace(' ', '')) > 498:
            fact = sample['fact_cut'][:498].replace(' ', '')
        else:
            fact = sample['fact_cut'].replace(' ', '')
        fact_text.append(fact)
        law_label.append(sample['law'])
        accu_label.append(sample['accu'])
        term_label.append(sample['term'])
    
    return fact_text, law_label, accu_label, term_label

class LkdfDataset(Dataset):
    def __init__(self, fact, law, accu, term, pad_idx = 0):
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-lert-base", do_lower_case=True)
        self.pad_idx = pad_idx
        self.dataset = self.preprocess(fact, law, accu, term)
        self.device = "cuda"

    def preprocess(self, fact_texts, law_labels, accu_labels, term_labels):
        data = []
        fact_text_list = []

        for fact_text in fact_texts:
            fact_tokens = self.tokenizer(fact_text)
            fact_input_ids = fact_tokens['input_ids']
            fact_token_type_ids = fact_tokens['token_type_ids']
            fact_attention_mask = fact_tokens['attention_mask']
            fact_text_list.append((fact_input_ids, fact_token_type_ids, fact_attention_mask))

        for fact_text, law_label, accu_label, term_label in zip(fact_text_list, law_labels, accu_labels, term_labels):
            data.append((fact_text, law_label, accu_label, term_label))

        return data
    
    def __getitem__(self, idx):
        fact = self.dataset[idx][0] #fact_text
        law = self.dataset[idx][1] #law_label
        accu = self.dataset[idx][2] #accu_label
        term = self.dataset[idx][3] #term_label

        return [fact, law, accu, term]
    
    def __len__(self):
        return len(self.dataset)
    
    def collate_fn(self, batch):
        batch_fact = [item[0] for item in batch]
        batch_law = [item[1] for item in batch]
        batch_accu = [item[2] for item in batch]
        batch_term = [item[3] for item in batch]

        batch_len = len(batch_fact)  #batch_size
        # max_len = min(max([len(item[0]) for item in batch_fact]), 512)
        max_len = 500

        pad_fact = self.pad_idx * np.ones((batch_len, max_len))
        pad_fact_attention = self.pad_idx * np.ones((batch_len, max_len))

        for i in range(batch_len):
            cur_len = len(batch_fact[i][0])
            if cur_len <= max_len:
                pad_fact[i][:cur_len] = batch_fact[i][0]
                pad_fact_attention[i][:cur_len] = batch_fact[i][2]
            else:
                pad_fact[i] = batch_fact[i][0][:max_len]
                pad_fact_attention[i] = batch_fact[i][2][:max_len]

        batch_fact = torch.tensor(pad_fact, dtype=torch.long)
        batch_fact_attention = torch.tensor(pad_fact_attention, dtype=torch.long)
        batch_law = torch.tensor(batch_law, dtype=torch.long)
        batch_accu = torch.tensor(batch_accu, dtype=torch.long)
        batch_term = torch.tensor(batch_term, dtype=torch.long)

        batch_fact = batch_fact.to(self.device)
        batch_fact_attention = batch_fact_attention.to(self.device)
        batch_law = batch_law.to(self.device)
        batch_accu = batch_accu.to(self.device)
        batch_term = batch_term.to(self.device)

        return [batch_fact, batch_fact_attention, batch_law, batch_accu, batch_term]
    
if __name__ == "__main__":
    fact, law, accu, term = load_data("./test_data/train.json")
    print(len(fact), len(law), len(accu), len(term))
    test_dataset = LkdfDataset(fact, law, accu, term)
    test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False, num_workers = 0, collate_fn = test_dataset.collate_fn, drop_last = True)
    #print(len(test_loader))
    for idx, batch_samples in enumerate(test_loader):
        print(idx, batch_samples)