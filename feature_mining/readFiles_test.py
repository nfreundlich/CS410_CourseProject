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

with open('model_feature.data', 'wb') as filehandle:
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

with open('model_feature.data', 'rb') as filehandle:
    # store the data as binary data stream
    rd_model_topic = pickle.load(filehandle)
    filehandle.close()

with open('docCounts.data', 'rb') as filehandle:
    # store the data as binary data stream
    rd_docCounts = pickle.load(filehandle)
    filehandle.close()


def format_feature_list(feature_list: list) -> pd.DataFrame:
    """
    This function takes a list of strings and/or lists of strings and converts them to a DataFrame with ids. Terms in
    nested lists will be treated as synonyms and given the same feature id

    ex. format_feature_list(feature_list = ["sound", "battery", ["screen", "display"]])

    :param feature_list: a list of strings and lists of strings. Individual strings will be given separate ids, lists
    of strings will be treated as synonyms and given the same feature id.
    ex. ["sound", "battery", ["screen", "display"]]
    :return: DataFrame with integer ids for each feature, synonyms are grouped together
    | feature (str) | feature_id (int)  | feature_term_id (int)
    feature: string representation of the feature
    feature_id: integer id for the feature, will be the same for synonyms if input in nested list
    feature_term_id: integer id for the feature, will be unique for each string, including synonyms

    """

    feature_index = 0
    feature_term_index = 0
    formatted_feature_list = []

    # loop through list of features
    for feature in feature_list:
        if isinstance(feature, str):
            print("string")
            formatted_feature_list.append(
                {"feature_term_id": feature_term_index, "feature_id": feature_index, "feature": feature})
            feature_term_index+=1
        elif isinstance(feature, list):
            print('list')
            for synonym in feature:
                if isinstance(synonym, str):
                    print('>string')
                    formatted_feature_list.append(
                        {"feature_term_id": feature_term_index, "feature_id": feature_index, "feature": synonym})
                    feature_term_index += 1
                else:
                    raise ValueError(str(feature) + '>' + str(synonym) + ' is not a string or a list of strings')

        else:
            raise ValueError(str(feature)+ ' is not a string or a list of strings')

        feature_index += 1

    feature_df = pd.DataFrame(formatted_feature_list)

    return feature_df


def read_annotated_data(filename: str, nlines: int =None) -> dict:
    """
    Reads in Santu's annotated files and records the explicit features and implicit features annotated in the file

    ex. annotated_data = read_annotated_data(filename='demo_files/iPod.final', nlines=200)

    :param filename: Filename for the annotated data set
    :param nlines: Maximum number of lines from the file to read or None to read all lines
    :return: a dictionary with the following data
        section_list: DataFrame with the following form
            | doc_id (int)  | section_id (int)  | section_text (str)    | title (bool)  |
            doc_id: integer id for the document
            section_id: integer id for the section
            section_text: cleaned (lowercase, trimmed) section text
            title: True if the line is a title, False otherwise
        feature_section_mapping: DataFrame
            | doc_id (int)  | feature (str) | is_explicit (bool)    | section_id (int)  |
            doc_id: integer id for the document
            feature: the string form of the feature in the annotation
            is_explicit: False if the feature was marked in the annotation as an implicit mention, True otherwise
            section_id: integer id for the section
        feature_list: dictionary with each feature and the number of sections it appears in
            key: feature name
            value: number of sections in which the feature appears
    """

    doc_id=-1
    section_id=0
    section_list = []
    feature_section_mapping = []
    feature_list = defaultdict(int)

    with open(filename,'r') as input_file:
        for line in input_file:

            print(line)

            # Section is from new doc, increment doc id
            if '[t]' in line:
                doc_id += 1
                is_title=True
                line_text = line.split('[t]')[1].strip().lower()

            # Section is from new doc, increment doc id
            elif line.startswith('*'):
                doc_id += 1
                is_title=True
                line_text = line.split('*')[1].strip().lower()

            # Section not from new doc, just get cleaned text
            else:
                is_title=False
                line_text=line.split('##')[1].strip().lower()

            # Look for feature annotations attached to the line
            feature_string = line.split('##')[0].split(',')
            print(feature_string)
            if not is_title and feature_string[0] != '':

                # Loop through all the features found in the annotation
                for feature in feature_string:
                    # print(feature)

                    # Check if the feature in the annotation is marked as an implicit mention
                    if '[u]' in feature:
                        explicit_feature=False
                        # print('implicit')
                    else:
                        explicit_feature=True

                    # Get the actual text of the feature
                    feature_text = feature.split('[@]')[0]

                    # Add the feature and section id to the data set
                    feature_section_mapping.append({"doc_id": doc_id, "section_id": section_id, "feature": feature_text,
                                                    "is_explicit": explicit_feature})

                    # Increment the feature in the unique feature list
                    feature_list[feature_text] += 1

            # Add section line to data set
            section_list.append(
                {"doc_id": doc_id, "section_id": section_id, "section_text": line_text, "title": is_title})

            # Increment section id
            section_id += 1
            # print(line)

            # Check if max number of lines has been reached yet
            if nlines is not None:
                if section_id >= nlines:
                    break

    # Bundle and return data set
    return dict(section_list=pd.DataFrame(section_list), feature_mapping=pd.DataFrame(feature_section_mapping),
                feature_list=feature_list)


def build_explicit_models(text_set: pd.DataFrame, feature_set: pd.DataFrame) -> dict:
    """
    This function builds a background model, set of topic models and summarizes the counts of words in each sentence
        to prepare for EM optimization

        ex. em_input = build_explicit_models(text_set=text_set, feature_set=feature_set)

    :param text_set: a pandas DataFrame with (at a minimum) the following columns
    :param feature_set: output of format feature list - OR -
        DataFrame with integer ids for each feature, synonyms are grouped together
        | feature (str) | feature_id (int)  | feature_term_id (int)
        feature: string representation of the feature
        feature_id: integer id for the feature, will be the same for synonyms if input in nested list
        feature_term_id: integer id for the feature, will be unique for each string, including synonyms
    :return: a dictionary with three entries -
        model_background: background model estimated from the entire document collection as described in section 4.2
        model_feature: feature models estimated from explicit mention sections as described in section 4.2
        section_word_counts: word counts in each section as needed by the EM algorithm
    """
    section_word_list = dict()  # list of all words in each section
    section_word_counts = dict()  # count of words in each section
    collection_word_counts = Counter()  # count of all words in all section
    word_section_counter = Counter()  # count of number of sections with word
    feature_word_counter = defaultdict(Counter) # keep track of words appearing in section with explicit feature mention
    feature_section_mapping = [] # keeps a list of the sentence ids associated with each feature (many-to-many mapping)

    # loop over all rows in input data set
    for index, row in text_set.iterrows():
        # print the current text for debugging
        print(str(row["sentence_id"]) + ":" + row["text"])

        # input the sentence into Spacy
        section = nlp(row["text"])

        # add each parsed word into a list
        # Note: won't catch capitalized stopwords, need to lowercase as part of pre-processing - also possible stop word
        # filtering not necessary because of tfidf term in topic model?
        current_section_words = []
        for word in section:
            if not word.is_stop and not word.is_punct:
                current_section_words.append(word.lemma_ if word.lemma_ != '-PRON-' else word.lower_)

        # get a count of distinct words in the section - this might need to be switched to default dict later
        current_section_word_counts = Counter(current_section_words)

        # get keys for distinct words to add to idf counter
        word_section_counter.update(current_section_word_counts.keys())

        # add these counts to the all section counter
        collection_word_counts.update(current_section_words)

        # add to section counts dictionary
        section_word_counts[row["sentence_id"]] = current_section_word_counts

        # add to dictionary holding word parsing
        section_word_list[row["sentence_id"]] = current_section_words

        # initialize list to keep track of found features (in case of synonyms)
        found_features = set()

        # get all explicit topics for this sentence and add these words to the list
        for index_f, row_f in feature_set.iterrows():

            # word was found in the section, record find and add words to feature topic model
            if row_f["feature"] in current_section_words:
                print("feature " + str(row_f["feature_id"]))

                if row_f("feature_id") in found_features:
                    # already found explicit feature mention in sentence as synonym, skip
                    continue
                else:
                    # feature has not been found yet, add to the list
                    found_features.add(row_f("feature_id"))

                # record that feature was explicitly found
                feature_section_mapping.append({"sentence_id": row["sentence_id"] ,"feature_id": row_f["feature_id"]})

                # if we only count each word once
                feature_word_counter[row_f["feature_id"]].update(current_section_word_counts.keys())

                # if we count each words as many times as it occurs
                # featureCounter[row_f["feature_id"]].update(current_section_words)

    # At this point we have all the counts we need to build the topic models

    ####################################
    # Calculations for background model
    ####################################

    # total number of words
    vocabulary_size = sum(collection_word_counts.values())

    # change counter to dictionary for calculations
    collection_word_counts = dict(collection_word_counts)

    # calculate background model
    model_background = dict((k, v / vocabulary_size) for k, v in collection_word_counts.items())

    ###############################
    # Calculations for topic model
    ###############################
    tfidf_feature = defaultdict(dict)
    model_feature_norms = Counter()

    # count of sentences
    section_count = len(section_word_list)
    # num_words = len(collection_word_counts)

    for word in collection_word_counts.keys():
        print(word)

        for index, row in feature_set.iterrows():
            print(str(index) + "-" + row["feature"])

            ##########################################################################
            # Formula 4, section 4.2, using base 2 logs, also adds +1 from Formula 5
            #########################################################################
            tfidf = math.log(1 + feature_word_counter[index][word], 2) \
                * math.log(1 + section_count / word_section_counter[word], 2) \
                + 1
            print(str(tfidf))

            tfidf_feature[index][word] = tfidf

            model_feature_norms[index] += tfidf

    # normalize values of all dictionaries with totals
    model_feature = defaultdict(dict)

    for index in model_feature_norms.keys():
        print("normalizing " + str(index))

        #########################################################################################################
        # Formula 5, section 4.2, using base 2 logs, +1 in numerator already taken care of in tfidf calculation
        #########################################################################################################
        model_feature[index] = dict((k, v / (model_feature_norms[index])) for k, v in tfidf_feature[index].items())

    model_results = dict(model_background=model_background, model_feature=model_feature, doc_counts=section_word_counts)

    return model_results



