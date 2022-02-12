from torch.utils.data import Dataset
from torchvision import transforms
from src.utils import Arxiv_preprocess
import json
from rouge import Rouge
import numpy as np
from tqdm import tqdm

class Section():
    def __init__(self, sentences:list, title, sigma=1):
        self.sentences = sentences
        self.fullSentences = ' '.join([s.sentence for s in self.sentences])
        self.title = title
        self.__normal = lambda x,mu: np.exp( - (x - mu)**2 / (2 * sigma**2) )
    
    def trueImportance(self, abstract, section_id, tot_sections):
        rouge = Rouge()
        relative_section_pos = section_id/tot_sections
        tot_score = 0
        for i,abs_sent in enumerate(abstract.sentences):
            relative_abs_sent_pos = i/len(abstract.sentences)
            weight = self.__normal(relative_abs_sent_pos,relative_section_pos)
            sent_score = rouge.get_scores(abs_sent.sentence,self.fullSentences)[0]['rouge-2']['r']
            tot_score += sent_score*weight
        self.trueImportance = tot_score
    
    def __iter__(self):
        yield 'title', self.title
        yield 'sentences', [s.sentence for s in self.sentences]

    def __len__(self) -> int:
        return sum([len(s) for s in self.sentences])
   
 
class Sentence():
    def __init__(self, sentence:str, padded_sentence:list, sigma=1):
        self.sentence = sentence
        self.padded_sentence = padded_sentence
        self.__normal = lambda x,mu: np.exp( - (x - mu)**2 / (2 * sigma**2) )
 
    def trueImportance(self, abstract:Section, section_id, tot_sections):
        rouge = Rouge()
        relative_section_pos = section_id/tot_sections
        tot_score = 0
        for i,abs_sent in enumerate(abstract.sentences):
            relative_abs_sent_pos = i/len(abstract.sentences)
            weight = self.__normal(relative_abs_sent_pos,relative_section_pos)
            rouge_scores = rouge.get_scores(abs_sent.sentence,self.sentence)[0]
            rouge_scores = np.mean([rouge_scores['rouge-1']['f'],rouge_scores['rouge-2']['f'],rouge_scores['rouge-l']['f']])
            tot_score += rouge_scores*weight
        self.trueImportance = tot_score

    def __len__(self) -> int:
        return len(self.sentence)


class Document():
    def __init__(self, sections:list):
        self.sections = sections
    
    def __iter__(self):
        for section in self.sections:
            #yield dict(section)
            yield section
    
    def __getitem__(self, item):
        return self.sections[item]

    def __len__(self) -> int:
        return sum(len(S) for S in self.sections)
    

class DynamicImportanceContentRankingDataset(Dataset):
    def __init__(self, data_path:str, params:dict, tokenizer, validation_set=False, pre_trained_tokenizer = None) -> None:
        '''
        json format for each document:
        { 
            'article_id': str,
            'abstract_text': List[str],
            'article_text': List[str],
            'section_names': List[str],
            'sections': List[List[str]]
        }
        '''
        f = open(f'./data/{data_path}','r')
        data = json.load(f)
        f.close()
        self.groundtruth = []

        for doc in data[:4]:
            self.groundtruth.append(' '.join(doc['abstract_text']))

        padding = params['padding']
        if not validation_set:
            self.tokenizer = tokenizer['class'](**tokenizer['params'])
            for doc in tqdm(data[:4],desc='Preparing Tokenizer'):
                for sentence in doc['abstract_text']:
                    self.tokenizer.add_sentence(sentence)
                for section in doc['sections']:
                    for sentence in section:
                        self.tokenizer.add_sentence(sentence)
                for section_title in doc['section_names']:
                    self.tokenizer.add_sentence(section_title)
            self.tokenizer.fit_tokenizer()
        else:
            self.tokenizer = pre_trained_tokenizer

        self.documents = []
        for doc in tqdm(data[:4],desc='Reading documents'):
            #Abstract
            sentenceList = []
            for sentence in doc['abstract_text']:
                sentenceList.append(Sentence(sentence,self.tokenizer.get_padded_sentence(sentence)))
            abstract = Section(sentenceList,None)

            sectionList = []
            for i,section in enumerate(doc['sections']):
                title = doc['section_names'][i]
                sentenceList = []
                for sentence in section:
                    s = Sentence(sentence,self.tokenizer.get_padded_sentence(sentence),params['sigma'])
                    s.trueImportance(abstract,i,len(doc['sections']))
                    sentenceList.append(s)

                sec = Section(sentenceList,Sentence(title,self.tokenizer.get_padded_sentence(title)),params['sigma'])
                sec.trueImportance(abstract,i,len(doc['sections']))
                sectionList.append(sec)

            self.documents.append(Document(sectionList))

    def __len__(self) -> int:
        #return sum([len(d) for d in self.documents])
        return len(self.documents)

    def __getitem__(self, index:int) -> Document:
        return self.documents[index]
