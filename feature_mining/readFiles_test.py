import pandas as pd
import spacy
from collections import Counter
from spacy.attrs import LOWER,ORTH
import en_core_web_sm

text_set = pd.read_csv('demo_files/sample_dataset_1_text.csv', skip_blank_lines=True, keep_default_na=False)

feature_set = pd.read_csv('demo_files/sample_dataset_1_features.csv', skip_blank_lines=True)

feature_mapping = pd.read_csv('demo_files/sample_dataset_1_feature_mapping.csv', skip_blank_lines=True)

print(text_set.head())
print(feature_set.head())
print(feature_mapping.head())

nlp = en_core_web_sm.load()
doc = nlp(u"this isn't a Test Doc or is it a test doc?")
docList = [word.lower_ for word in doc]
words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]

counts = doc.count_by(ORTH)
print(len(counts))
for word_id, count in sorted(counts.items(), reverse=True, key=lambda item: item[1]):
    print(count, nlp.vocab.strings[word_id])

docList = dict() # list of all terms in each sentence
docCounts = dict() # count of terms in each sentence
for index,row in text_set.iterrows():
    print(row["text"])
    doc = nlp(row["text"])
    docWordList = [word.lower_ for word in doc]
    wordCounts = Counter(docWordList)
    docCounts[row["sentence_id"]] = wordCounts
    docList[row["sentence_id"]] = docWordList

# count of sentences
len(docList)
