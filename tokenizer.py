import nltk

from qa_utils.text import Tokenizer


class NLTKTokenizer(Tokenizer):
    """
    Applies lower case folding and uses nltk's `word_tokenize(text)`.
    """

    def tokenize(self, text):
        return nltk.word_tokenize(text.lower())
