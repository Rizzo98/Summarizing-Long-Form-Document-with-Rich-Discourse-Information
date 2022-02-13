from torch.utils.data import Dataset
from torchvision import transforms
from src.tokenizers import ContentRankingTokenizer
import json
from rouge import Rouge
import numpy as np
from tqdm import tqdm

class Section():
    def __init__(self, sentences:list, title):
        self.sentences = sentences
        self.fullSentences = ' '.join([s.sentence for s in self.sentences])
        self.title = title
    
    def trueImportance(self, abstract):
        rouge = Rouge()
        rouge_scores = rouge.get_scores(abstract.fullSentences,self.fullSentences)[0]
        self.trueImportance = rouge_scores['rouge-2']['r']
    
    def __iter__(self):
        yield 'title', self.title
        yield 'sentences', [s.sentence for s in self.sentences]

    def __len__(self) -> int:
        return sum([len(s) for s in self.sentences])
   
 
class Sentence():
    def __init__(self, sentence:str, padded_sentence:list):
        self.sentence = sentence
        self.padded_sentence = padded_sentence
 
    def trueImportance(self, abstract:Section):
        rouge = Rouge()
        rouge_scores = rouge.get_scores(abstract.fullSentences,self.sentence)[0]
        self.trueImportance = np.mean([rouge_scores['rouge-1']['f'],rouge_scores['rouge-2']['f'],rouge_scores['rouge-l']['f']])

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
    

class DocumentDataset(Dataset):
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
                    s = Sentence(sentence,self.tokenizer.get_padded_sentence(sentence))
                    s.trueImportance(abstract)
                    sentenceList.append(s)

                sec = Section(sentenceList,Sentence(title,self.tokenizer.get_padded_sentence(title)))
                sec.trueImportance(abstract)
                sectionList.append(sec)

            self.documents.append(Document(sectionList))

    def __len__(self) -> int:
        #return sum([len(d) for d in self.documents])
        return len(self.documents)

    def __getitem__(self, index:int) -> Document:
        return self.documents[index]
