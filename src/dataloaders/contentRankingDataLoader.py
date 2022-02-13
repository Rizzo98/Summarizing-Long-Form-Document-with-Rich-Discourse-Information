import numpy as np
import torch
from torch.utils.data import DataLoader

class ContentRankingDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=2, shuffle=True, padding=30, max_sections=4, max_sentences_per_section=10, device='cpu'):
        super(ContentRankingDataLoader,self).__init__(dataset, batch_size, shuffle, collate_fn = self.custom_collate)
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