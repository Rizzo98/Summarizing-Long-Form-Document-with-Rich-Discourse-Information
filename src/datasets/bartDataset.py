from datasets import Dataset
import json
class BartDataset(Dataset):
    def __init__(self, data_path:str, params:dict, tokenizer,
     validation_set=False, pre_trained_tokenizer = None, fromWrapper=False, datasetFromWrapper = None):
        
        self.tokenizer = tokenizer['class'](**tokenizer['params'])
        self.raw_docs = None

        if not fromWrapper:
            f = open(f'./data/{data_path}','r')
            data = json.load(f)
            f.close()
            d = {'Summary':[], 'Text':[], 'Doc_id':[]}
            for doc_id,doc in enumerate(data):
                abstract = ' '.join(doc['abstract_text'])
                for sentences_list in doc['sections']:
                    sentences_concat = ' '.join(sentences_list)
                    d['Summary'].append(abstract)
                    d['Text'].append(sentences_concat)
                    d['Doc_id'].append(doc_id)
            self.dataset = Dataset.from_dict(d)
        else:
            self.dataset = datasetFromWrapper


        inputs = self.tokenizer(self.dataset["Text"])
        outputs = self.tokenizer(self.dataset["Summary"])

        self.dataset = self.dataset.add_column("input_ids", inputs.input_ids)
        self.dataset = self.dataset.add_column("attention_mask", inputs.attention_mask)
        self.dataset = self.dataset.add_column("decoder_input_ids", outputs.input_ids)
        self.dataset = self.dataset.add_column("decoder_attention_mask", outputs.attention_mask)
        self.dataset = self.dataset.add_column("labels",[[-100 if token == self.tokenizer.tokenizer.pad_token_id else token for token in labels] for labels in outputs.input_ids.copy()])
        
        self.dataset = self.dataset.remove_columns(["Text", "Summary"])

    def __call__(self):
        return self.dataset
