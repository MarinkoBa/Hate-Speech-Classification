import spacy
import re

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner'])
stopwords = nlp.Defaults.stop_words
stemmer = PorterStemmer()


def preprocessing(text):
    """
        Applies different steps of preprocessing to a text.
        Preprocessing includes:

        - remove non-standard lexical tokens (which are not numeric or alphabetical)
        - remove url and @name mentions
        - remove standard stopwords (english stopwords from spacy)
        - convert all letters to lower case
        - perform stemming (PorterStemmer nltk)
        
        Parameters
        ----------
        text:      String
                        Text which should be converted by preprocessing

        Returns
        -------
        preprocessed_text:  String
                        Text which is converted by preprocessing.
    """

    # remove (twitter) urls
    text = re.sub(r"http://t.co/[a-zA-Z0-9]+", "", text)

    # remove all @name Mentions (Usernames only allowed to includes characters A-Z, 0-9 and underscores)
    text = re.sub(r"@[a-zA-Z0-9_]+", "", text)

    # remove non alphabetical characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # convert all letters to lower case
    text = text.lower()

    # split text to single words
    words = word_tokenize(text)
    tokens = []

    # remove stopwords and words with length 1
    for word in words:
        if word not in stopwords:
            if len(word) > 1:
                # apply stemmer to single word
                word = stemmer.stem(word)
                tokens.append(word)

    # convert tokens back to text
    preprocessed_text = ' '.join([str(element) for element in tokens])

    return preprocessed_text
