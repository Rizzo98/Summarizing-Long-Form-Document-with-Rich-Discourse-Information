import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import TrainingArguments, Seq2SeqTrainingArguments
from dataclasses import dataclass, field
from typing import Optional

class ArxivDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=2, shuffle=True, padding=30, max_sections=4, max_sentences_per_section=10, device='cpu'):
        super(ArxivDataLoader,self).__init__(dataset, batch_size, shuffle, collate_fn = self.custom_collate)
        self.padding = padding
        self.max_sections = max_sections
        self.max_sentences = max_sentences_per_section
        self.device = device

    def custom_collate(self, batch: list):
        data = {
            'titles': torch.zeros((len(batch),self.max_sections,self.padding),device=self.device),
            'sections': torch.zeros((len(batch),self.max_sections,self.max_sentences,self.padding),device=self.device)}
        section_importance = torch.zeros((len(batch),self.max_sections),device=self.device)
        sentence_importance = torch.zeros((len(batch),self.max_sections,self.max_sentences),device=self.device)

        for doc_id,doc in enumerate(batch):
            titles = [ np.zeros(self.padding) for _ in range(self.max_sections)]
            sections = [ [np.zeros(self.padding) for _ in range(self.max_sentences)] for _ in range(self.max_sections)]
            for i,sec in enumerate(doc[:self.max_sections]):
                section_importance[doc_id][i] = torch.tensor(sec.trueImportance, device=self.device)
                titles[i] = sec.title.padded_sentence
                sentences = [ np.zeros(self.padding) for _ in range(self.max_sentences) ]
                for j,sent in enumerate(sec.sentences[:self.max_sentences]):
                    sentence_importance[doc_id][i][j]= torch.tensor(sent.trueImportance, device=self.device)
                    sentences[j] = sent.padded_sentence
                sections[i] = sentences

            titles = torch.tensor(titles, device=self.device)
            sections = torch.tensor(sections, device=self.device)
            data['titles'][doc_id] = titles
            data['sections'][doc_id] = sections
        section_importance = torch.unsqueeze(section_importance, 2)
        return data, section_importance, sentence_importance

class BartDataLoader():
    def __init__(self, dataset, batch_size=32, encoder_max_length=100, decoder_max_length=20,
     num_train_epochs=3, device='cpu', shuffle=False, outputFolder=''):
        self.tokenizer = dataset.tokenizer
        self.raw_docs = dataset.raw_docs
        self.dataset = dataset().map( 
            lambda x:x,
            batched=True, 
            batch_size=batch_size
        )
        self.dataset.set_format(
            type="torch", columns=['Doc_id', "input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        )
        self.args = Seq2SeqTrainingArguments(
            output_dir=f"./outputs/{outputFolder}/",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            predict_with_generate=True,
            evaluation_strategy="epoch",
            do_train=True,
            do_eval=True,
            logging_steps=2, 
            save_steps=16, 
            eval_steps=500, 
            warmup_steps=500, 
            overwrite_output_dir=True,
            save_total_limit=1,
            report_to='none'
            )
        
    def __iter__(self):
        yield tuple(self.dataset[0].values())
    
    def __len__(self):
        return len(self.dataset)