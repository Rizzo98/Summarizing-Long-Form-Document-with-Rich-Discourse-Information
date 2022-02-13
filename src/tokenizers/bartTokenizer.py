from transformers import RobertaTokenizerFast

class BartTokenizer(RobertaTokenizerFast):
    def __init__(self, padding="max_length", truncation=True, max_length=100):
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    
    def __call__(self, doc):
        return self.tokenizer(doc,padding=self.padding, truncation=self.truncation, max_length=self.max_length)