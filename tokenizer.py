import nltk as nltk


class NLTKTokenizer:
    @staticmethod
    def tokenize(text):
        return nltk.word_tokenize(text.lower())
