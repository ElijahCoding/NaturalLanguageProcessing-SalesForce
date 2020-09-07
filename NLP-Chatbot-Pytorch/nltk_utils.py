import nltk

nltk.download('punkt')

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    pass

def bag_of_words(tokenized_sentence, all_words):
    pass