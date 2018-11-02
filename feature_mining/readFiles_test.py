import pandas as pd
import spacy
from collections import Counter
from collections import defaultdict
from spacy.attrs import LOWER,ORTH
import en_core_web_sm

# Reference for counter speed: http://evanmuehlhausen.com/simple-counters-in-python-with-benchmarks/

text_set = pd.read_csv('demo_files/sample_dataset_1_text.csv', skip_blank_lines=True, keep_default_na=False)
text_set = text_set[text_set['sentence_id'] != '']

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
docCounts_All=Counter() # count of all words in all sentences
sentenceCounter = Counter() # count of number of sentences with word
featureCounter = defaultdict(Counter)

# loop over all rows in input data set
for index,row in text_set.iterrows():
    # print the current text for debugging
    print(str(row["sentence_id"]) + ":" + row["text"])

    # input the sentence into Spacy
    doc = nlp(row["text"])

    # add each parsed word into a list via list comprehension
    docWordList = [word.lower_ for word in doc]

    # get a count of distinct words in the doc - this might need to be switched to default dict later
    wordCounts = Counter(docWordList)

    # get keys for distinct words to add to idf counter
    sentenceCounter.update(wordCounts.keys())

    # add these counts to the all doc counter
    docCounts_All.update(docWordList)

    # add to doc counts dictionary
    docCounts[row["sentence_id"]] = wordCounts

    # add to dictionary holding word parsing
    docList[row["sentence_id"]] = docWordList

    # get all explicit topics for this sentence and add these words to the list
    sentence_features = feature_mapping.loc[feature_mapping['sentence_id'] == int(row["sentence_id"])]
    if len(sentence_features.index) > 0:
        for index_f,row_f in sentence_features.iterrows():
            print("feature " + str(row_f["feature_id"]))

            # if we only count each word once
            featureCounter[row_f["feature_id"]].update(wordCounts.keys())

            # if we count each words as many times as it occurs
            # featureCounter[row_f["feature_id"]].update(docWordList)




# count of sentences
len(docList)
