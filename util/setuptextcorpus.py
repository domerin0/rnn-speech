try:
    from nltk.corpus import reuters
except ImportError:
    print("nltk not installed, use 'pip install nltk'")

def get_corpus_text():
    '''
    return raw text of reuters corpus
    '''
    return [" ".join(reuters.words(fid)) for fid in reuters.fileids()]
