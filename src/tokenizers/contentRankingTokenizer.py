from keras import preprocessing

class ContentRankingTokenizer:
    def __init__(self, padding:int):
        self.tokenizer = preprocessing.text.Tokenizer()
        self.padding = padding
        self.sentences = []
    
    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def fit_tokenizer(self):
        assert len(self.sentences)>0, 'sentences list is empty!'
        self.tokenizer.fit_on_texts(self.sentences)
        self.vocab_length = len(self.tokenizer.word_index) + 1

    def get_padded_sentence(self, sentence:str):
        return preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([sentence]), self.padding, padding='post')[0]
    