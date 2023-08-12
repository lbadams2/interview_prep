from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus) # converts a collection of text documents into a matrix of token counts
vectorizer.get_feature_names_out()
print(X.toarray())

vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2)) # (1,1) only unigrams, (2,2) only bigrams, (1,2) both uni and bi
X2 = vectorizer2.fit_transform(corpus)
vectorizer2.get_feature_names_out()
print(X2.toarray())