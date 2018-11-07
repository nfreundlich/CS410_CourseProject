import pandas as pd
import spacy
from collections import Counter
from collections import defaultdict
from spacy.attrs import LOWER, ORTH
import en_core_web_sm
import math
import pickle
import os

# Change working directory
os.chdir('C:\\Users\\Project Code\\feature_mining')

# Reference for counter speed: http://evanmuehlhausen.com/simple-counters-in-python-with-benchmarks/

text_set = pd.read_csv('demo_files/sample_dataset_1_text.csv', skip_blank_lines=True, keep_default_na=False)
# text_set = text_set[text_set['sentence_id'] != '']
text_set = text_set[text_set['sentence_id'].isin(['9', '10'])]

feature_set = pd.read_csv('demo_files/sample_dataset_1_features.csv', skip_blank_lines=True)
feature_set = feature_set[feature_set['feature_id'].isin([1, 2])]

feature_mapping = pd.read_csv('demo_files/sample_dataset_1_feature_mapping.csv', skip_blank_lines=True)

print(text_set.head())
print(feature_set.head())
print(feature_mapping.head())

nlp = en_core_web_sm.load()
doc = nlp(u"GOING going this isn't a Test Doc or is it a test doc?")
docList = [word.lemma_ if not word.is_stop else 'stopword' for word in doc]
docList = [word.lemma_ for word in doc]
words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]

counts = doc.count_by(ORTH)
print(len(counts))
for word_id, count in sorted(counts.items(), reverse=True, key=lambda item: item[1]):
    print(count, nlp.vocab.strings[word_id])

docList = dict()  # list of all terms in each sentence
docCounts = dict()  # count of terms in each sentence
docCounts_All = Counter()  # count of all words in all sentences
sentenceCounter = Counter()  # count of number of sentences with word
featureCounter = defaultdict(Counter)

# loop over all rows in input data set
for index, row in text_set.iterrows():
    # print the current text for debugging
    print(str(row["sentence_id"]) + ":" + row["text"])

    # input the sentence into Spacy
    doc = nlp(row["text"])

    # add each parsed word into a list via list comprehension
    docWordList = [word.lower_ if not word.is_stop else 'stopword' for word in doc]

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
        for index_f, row_f in sentence_features.iterrows():
            print("feature " + str(row_f["feature_id"]))

            # if we only count each word once
            featureCounter[row_f["feature_id"]].update(wordCounts.keys())

            # if we count each words as many times as it occurs
            # featureCounter[row_f["feature_id"]].update(docWordList)

# At this point we have all the counts we need to build the topic models

####################################
# Calculations for background model
####################################

# total number of words
totalWordCount = sum(docCounts_All.values())

# change counter to dictionary for calculations
docCounts_All_dict = dict(docCounts_All)

# calculate background model
model_background = dict((k, v / totalWordCount) for k, v in docCounts_All_dict.items())

###############################
# Calculations for topic model
###############################
tfidf_topic = defaultdict(dict)
model_topic_norms = Counter()

# count of sentences
numSentences = len(docList)
numWords = len(docCounts_All_dict)

for word in docCounts_All_dict.keys():
    print(word)

    for index, row in feature_set.iterrows():
        print(str(index) + "-" + row["feature"])

        tfidf = math.log(1 + featureCounter[index][word]) * math.log(1 + numSentences / sentenceCounter[word]) + 1
        print(str(tfidf))

        tfidf_topic[index][word] = tfidf

        model_topic_norms[index] += tfidf

# normalize values of all dictionaries with totals
model_topic = defaultdict(dict)

for index in model_topic_norms.keys():
    print("normalizing " + str(index))

    model_topic[index] = dict((k, v / (model_topic_norms[index])) for k, v in tfidf_topic[index].items())

# list data that needs to be passed to to EM Algorithm

# save data into file for Norbert to load up

with open('model_background.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(model_background, filehandle)
    filehandle.close()

with open('model_topic.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(model_topic, filehandle)
    filehandle.close()

with open('docCounts.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(docCounts, filehandle)
    filehandle.close()

# test reload

with open('model_background.data', 'rb') as filehandle:
    # store the data as binary data stream
    rd_model_background = pickle.load(filehandle)
    filehandle.close()

with open('model_topic.data', 'rb') as filehandle:
    # store the data as binary data stream
    rd_model_topic = pickle.load(filehandle)
    filehandle.close()

with open('docCounts.data', 'rb') as filehandle:
    # store the data as binary data stream
    rd_docCounts = pickle.load(filehandle)
    filehandle.close()


def build_explicit_models(text_set: object, feature_set: object, feature_mapping: object) -> object:
    """
    This function builds a background model, set of topic models and summarizes the counts of words in each sentence
        to prepare for EM optimization

    :param text_set: a pandas dataframe with (at a minimum) the following columns
    :param feature_set:
    :param feature_mapping:
    :return: a dictionary with three entries -
        model_background: background model estimated from the entire document collection as described in section 4.2
        model_topic: topic models estimated from explicit mention sentences as described in section 4.2
        doc_counts: word counts in each sentence as needed by the EM algorithm
    """
    doc_list = dict()  # list of all terms in each sentence
    doc_counts = dict()  # count of terms in each sentence
    doc_counts_all = Counter()  # count of all words in all sentences
    sentence_counter = Counter()  # count of number of sentences with word
    feature_counter = defaultdict(Counter)

    # loop over all rows in input data set
    for index, row in text_set.iterrows():
        # print the current text for debugging
        print(str(row["sentence_id"]) + ":" + row["text"])

        # input the sentence into Spacy
        doc = nlp(row["text"])

        # add each parsed word into a list via list comprehension
        doc_word_list = []
        for word in doc:
            if not word.is_stop and not word.is_punct:
                doc_word_list.append(word.lemma_ if word.lemma_ != '-PRON-' else word.lower_)

        # get a count of distinct words in the doc - this might need to be switched to default dict later
        word_counts = Counter(doc_word_list)

        # get keys for distinct words to add to idf counter
        sentence_counter.update(word_counts.keys())

        # add these counts to the all doc counter
        doc_counts_all.update(doc_word_list)

        # add to doc counts dictionary
        doc_counts[row["sentence_id"]] = word_counts

        # add to dictionary holding word parsing
        doc_list[row["sentence_id"]] = doc_word_list

        # get all explicit topics for this sentence and add these words to the list
        sentence_features = feature_mapping.loc[feature_mapping['sentence_id'] == int(row["sentence_id"])]
        if len(sentence_features.index) > 0:
            for index_f, row_f in sentence_features.iterrows():
                print("feature " + str(row_f["feature_id"]))

                # if we only count each word once
                feature_counter[row_f["feature_id"]].update(word_counts.keys())

                # if we count each words as many times as it occurs
                # featureCounter[row_f["feature_id"]].update(doc_word_list)

    # At this point we have all the counts we need to build the topic models

    ####################################
    # Calculations for background model
    ####################################

    # total number of words
    total_word_count = sum(doc_counts_all.values())

    # change counter to dictionary for calculations
    doc_counts_all_dict = dict(doc_counts_all)

    # calculate background model
    model_background = dict((k, v / total_word_count) for k, v in doc_counts_all_dict.items())

    ###############################
    # Calculations for topic model
    ###############################
    tfidf_topic = defaultdict(dict)
    model_topic_norms = Counter()

    # count of sentences
    num_sentences = len(doc_list)
    num_words = len(doc_counts_all_dict)

    for word in doc_counts_all_dict.keys():
        print(word)

        for index, row in feature_set.iterrows():
            print(str(index) + "-" + row["feature"])

            tfidf = math.log(1 + feature_counter[index][word]) * math.log(
                1 + num_sentences / sentence_counter[word]) + 1
            print(str(tfidf))

            tfidf_topic[index][word] = tfidf

            model_topic_norms[index] += tfidf

    # normalize values of all dictionaries with totals
    model_topic = defaultdict(dict)

    for index in model_topic_norms.keys():
        print("normalizing " + str(index))

        model_topic[index] = dict((k, v / (model_topic_norms[index])) for k, v in tfidf_topic[index].items())

    model_results = dict(model_background=model_background, model_topic=model_topic, doc_counts=doc_counts)

    return model_results


em_input = build_explicit_models(text_set=text_set, feature_set=feature_set, feature_mapping=feature_mapping)
