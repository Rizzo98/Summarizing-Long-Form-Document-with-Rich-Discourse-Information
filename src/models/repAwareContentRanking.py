import torch
from torch import nn
import torch.nn.functional as F
from src.datasets import *
from src.models.attention_mechanism.attention import Attention
from src.models.embedding.gloveEmbedding import Glove25Embedding

class RepAwareContentRanking(nn.Module):
    def __init__(self, tokenizer=None, embedding_dim=25, device='cpu'):
        super(RepAwareContentRanking, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = self.getEmbedder(embedding_dim)(tokenizer)
        
        self.titleEncoder = nn.LSTM(embedding_dim,embedding_dim,1,bidirectional=True, batch_first=True)
        self.titleAttention = Attention(embedding_dim)
        
        self.sentenceEncoder = nn.LSTM(embedding_dim,embedding_dim,1,bidirectional=True, batch_first=True)
        self.sentenceAttention = Attention(embedding_dim)

        self.sectionEncoder = nn.LSTM(embedding_dim*2, embedding_dim,1,bidirectional=True)
        self.sectionAttention = Attention(embedding_dim)

        self.sentenceImportance = nn.Linear(embedding_dim*2,1)
        self.sectionTitleAttention = Attention(embedding_dim)
        self.sectionTitleImportance = nn.Linear(embedding_dim*2,1)

        self.to(device)

    def getEmbedder(self, embedding_dim):
        embedders = {
            25: Glove25Embedding
            }
        assert embedding_dim in embedders.keys(), 'No embedder for that dimension.'
        return embedders[embedding_dim]

        
    def forward(self, batch_documents: list, device='cpu'):
        titles = batch_documents['titles']
        title_embeds = self.embedding(titles)
        title_embeds = torch.swapaxes(title_embeds,0,1) #title_embeds shape -> [section, document, title, embed]
        title_store = torch.zeros((title_embeds.shape[0], title_embeds.shape[1],title_embeds.shape[3]*2),device=device) #shape -> [doc, embed*2]
        for i,title_section in enumerate(title_embeds):
            title_embed, _ = self.titleEncoder(title_section)
            title_embed = self.titleAttention(title_embed)
            title_store[i] = title_embed
        sections = batch_documents['sections'] #2,4,100,30
        section_embeds = self.embedding(sections)
        sentences_importance = torch.zeros((section_embeds.size()[0],section_embeds.size()[1],section_embeds.size()[2]),device=device) # (doc,sect,sent)
        section_embeds = torch.swapaxes(section_embeds,0,1)
        section_embeds = torch.swapaxes(section_embeds,1,2) # section_embeds shape -> [section, sentence, doc, word]

        sections_importance = torch.zeros((section_embeds.shape[0],section_embeds.shape[2],1),device=device)
        sentence_store = torch.zeros((len(section_embeds),section_embeds[0].size()[0],section_embeds[0].size()[1],section_embeds[0].size()[3]*2),device=device)
        for section_id, section in enumerate(section_embeds):
            for i,sentence in enumerate(section):
                sentence_embeds, _ = self.sentenceEncoder(sentence)
                sentence_embeds = self.sentenceAttention(sentence_embeds)
                sentence_importance = torch.sigmoid(self.sentenceImportance(sentence_embeds))
                for doc_id, single_sentence_importance in enumerate(sentence_importance):
                    sentences_importance[doc_id][section_id][i] = single_sentence_importance
                sentence_store[section_id][i] = sentence_embeds
            section_embeds,_ = self.sectionEncoder(torch.swapaxes(sentence_store,2,1)[section_id])
            section_embeds = self.sentenceAttention(section_embeds)
            section_title_embed = torch.stack((title_store[section_id],section_embeds), dim=1)
            section_title_embed = self.sectionTitleAttention(section_title_embed)
            section_importance = torch.sigmoid(self.sectionTitleImportance(section_title_embed))
            sections_importance[section_id] = section_importance
        sections_importance = torch.swapaxes(sections_importance,0,1)
        sentence_store = torch.swapaxes(sentence_store,2,1)
        sentence_store = torch.swapaxes(sentence_store,1,0)
        return sections_importance, sentences_importance, sentence_store

    def load(self,model):
        self.load_state_dict(torch.load(model))
