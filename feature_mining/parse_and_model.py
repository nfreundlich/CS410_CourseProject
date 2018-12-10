import math
import logging
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import nltk
from nltk.tokenize import sent_tokenize
from spacy.cli.download import download as spacy_download
try:
    import en_core_web_sm
except ImportError:
    logging.warning(">Spacy  en_core_web_sm not found. Downloading and installing.")
    spacy_download("en_core_web_sm")
    import en_core_web_sm
from collections import Counter, defaultdict, OrderedDict
import time
import os
from enum import Enum, auto


class ParseAndModel:
    """
    Treats data input chain.
    Based on this data, computes matrices for reviews and features.
    Usage:
        pm = ParseAndModel(feature_list=["sound", "battery", ["screen", "display"]],
                       filename='../tests/data/parse_and_model/iPod.final')
        print(pm.model_results)
    """

    class InputType(Enum):
        """
        Enum holding the types of input files.
        """
        annotated = auto()
        docperline = auto()

    def __init__(self, feature_list: list = None,
                 filename: str = None,
                 input_type: Enum = InputType.annotated,
                 nlines: int = None,
                 remove_stopwords: bool = True,
                 start_line: int = 0,
                 lemmatize_words: bool = True,
                 log_base: int = None,
                 include_title_lines: bool = True):
        """


        :param feature_list: a list of strings and lists of strings. Individual strings will be given separate ids, lists
        of strings will be treated as synonyms and given the same feature id.
        ex. ["sound", "battery", ["screen", "display"]]
        :param filename: Filename for the data set
        :param input_type: A string specifying the type of input data so the correct read function can be chosen
            options are "annotated" - which expects data in Santu's original format and "oneDocPerLine" - which expects
            all data to be in a single file with one document per line
        :param nlines: Maximum number of lines from the file to read or None to read all lines
        :param remove_stopwords: Set to true if stop words should be removed from document sections before models are
            created
        :param start_line: Optional parameter, specific line number to start at, mostly for testing purposes
        :param lemmatize_words: Set to true if lemmatization should be performed on document sections before models are
            created
        :param log_base: Optional parameter to specify log base, defaults to ln if not set
        :param include_title_lines: Set to true to include lines as marked in title lines in the output, false otherwise
            only valid for annotated data input
        """
        # Test nltk dependencies
        nltk_punkt = nltk.data.find('tokenizers/punkt')
        if nltk_punkt is None:
            logging.warning(" >NLTK punkt not present: downloading nltk punkt.")
            nltk.download('punkt')
        else:
            logging.info(" >Ok: NLTK punkt present.")

        # Run feature list formatter and save output (or notify user this is being skipped)
        self.feature_list = feature_list
        self.formatted_feature_list = None
        if self.feature_list is None:
            logging.warning(" >No feature list specified, skipping feature list formatting")
        else:
            self.formatted_feature_list = self.format_feature_list()

        # Run read annotated data (or notify user this is being skipped)
        if filename is None:
            logging.warning(" >No filename specified, skipping parse step")
        else:
            if input_type == ParseAndModel.InputType.annotated:
                logging.info("Reading data from annotated file")
                self.parsed_text = self.read_annotated_data(filename=filename, nlines=nlines, start_line=start_line,
                                                            include_title_lines=include_title_lines)
            elif input_type == ParseAndModel.InputType.docperline:
                logging.info("Reading data from un-annotated file. Assuming one document per line.")
                self.parsed_text = self.read_file_data(filename=filename, nlines=nlines, start_line=start_line)

            else:
                raise Exception("Invalid input type. Options are 'annotated' and 'oneDocPerLine'")

        # Build the explicit models and store the output
        if self.formatted_feature_list is None:
            logging.warning(" >No formatted feature list present, can't build explicit models")
        elif self.parsed_text is None:
            logging.warning(" >No parsed text present, can't build explicit models")
        else:
            self.model_results = self.build_explicit_models(remove_stopwords=remove_stopwords,
                                                            lemmatize_words=lemmatize_words,
                                                            log_base=log_base
                                                            )

        # self.parsed_text2 = ParseAndModel.read_file_data(filename=filename, nlines=nlines, start_line=start_line)

    def format_feature_list(self) -> pd.DataFrame:
        """
        This function takes a list of strings and/or lists of strings and converts them to a DataFrame with ids. Terms in
        nested lists will be treated as synonyms and given the same feature id

        ex. feature_list = format_feature_list(feature_list = ["sound", "battery", ["screen", "display"]])

        :return: DataFrame with integer ids for each feature, synonyms are grouped together
        | feature (str) | feature_id (int)  | feature_term_id (int)
        feature: string representation of the feature
        feature_id: integer id for the feature, will be the same for synonyms if input in nested list
        feature_term_id: integer id for the feature, will be unique for each string, including synonyms

        """
        feature_list = self.feature_list

        feature_index = 0
        feature_term_index = 0
        formatted_feature_list = []

        # loop through list of features
        for feature in feature_list:
            if isinstance(feature, str):
                formatted_feature_list.append(
                    {"feature_term_id": feature_term_index, "feature_id": feature_index, "feature": feature})
                feature_term_index += 1
            elif isinstance(feature, list):
                for synonym in feature:
                    if isinstance(synonym, str):
                        formatted_feature_list.append(
                            {"feature_term_id": feature_term_index, "feature_id": feature_index, "feature": synonym})
                        feature_term_index += 1
                    else:
                        raise ValueError(str(feature) + '>' + str(synonym) + ' is not a string or a list of strings')

            else:
                raise ValueError(str(feature) + ' is not a string or a list of strings')

            feature_index += 1

        feature_df = pd.DataFrame(formatted_feature_list)

        # Save formatted feature list to object
        # TODO: [nfr] remove this from here, return feature_df and make assignment in __init__
        return feature_df

    # TODO: add tests, alterate file formats
    def read_annotated_data(self, filename: str, nlines: int = None, start_line: int = 0,
                            include_title_lines: bool = True) -> dict:
        """
        Reads in Santu's annotated files and records the explicit features and implicit features annotated in the file

        ex. annotated_data = read_annotated_data(filename='demo_files/iPod.final', nlines=200)
        ex. annotated_data = read_annotated_data(filename='demo_files/iPod.final', nlines=2)

       :param filename: Filename for the annotated data set
       :param nlines: Maximum number of lines from the file to read or None to read all lines
       :param start_line: Optional parameter, specific line number to start at, mostly for testing purposes
       :param include_title_lines: Set to true to include lines as marked in title lines in the output, false otherwise
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
        doc_id = -1
        section_id = 0
        section_list = []
        feature_section_mapping = []
        feature_list = defaultdict(int)
        line_number = 0
        line_count = 0

        with open(filename, 'r') as input_file:
            for line in input_file:

                # Skip line if before specified start
                if line_number < start_line:
                    # Increment line number
                    line_number += 1
                    continue
                else:
                    # Increment line number
                    line_number += 1

                # Section is from new doc, increment doc id
                if '[t]' in line:
                    doc_id += 1
                    is_title = True
                    line_text = line.split('[t]')[1].strip().lower()

                # Section is from new doc, increment doc id
                elif line.startswith('*'):
                    doc_id += 1
                    is_title = True
                    line_text = line.split('*')[1].strip().lower()

                # Section not from new doc, just get cleaned text
                else:
                    is_title = False
                    line_text = line.split('##')[1].strip().lower()

                # If we still haven't seen a title increment the document id anyway
                if doc_id == -1:
                    doc_id += 1

                # Look for feature annotations attached to the line
                feature_string = line.split('##')[0].split(',')
                logging.debug(feature_string)
                if not is_title and feature_string[0] != '':

                    # Loop through all the features found in the annotation
                    for feature in feature_string:
                        logging.debug(feature)

                        # Check if the feature in the annotation is marked as an implicit mention
                        if '[u]' in feature:
                            explicit_feature = False
                            logging.debug('implicit')
                        else:
                            explicit_feature = True

                        # Get the actual text of the feature
                        feature_text = feature.split('[@]')[0]

                        # Add the feature and section id to the data set
                        feature_section_mapping.append(
                            {"doc_id": doc_id, "section_id": section_id, "feature": feature_text,
                             "is_explicit": explicit_feature})

                        # Increment the feature in the unique feature list
                        feature_list[feature_text] += 1

                # Check if title lines should be included
                if not include_title_lines and is_title:
                    # Check if max number of lines has been reached yet
                    line_count += 1
                    if nlines is not None:
                        if line_count >= nlines:
                            break

                    continue

                # Add section line to data set
                section_list.append(
                    {"doc_id": doc_id, "section_id": section_id, "section_text": line_text, "title": is_title})

                # Increment section id
                section_id += 1
                line_count += 1
                logging.debug(line)

                # Check if max number of lines has been reached yet
                if nlines is not None:
                    if line_count >= nlines:
                        break

        # Bundle and save data set
        # TODO: [nfr] remove this from here, return dictionary and make assignment in __init__
        return dict(section_list=pd.DataFrame(section_list), feature_mapping=pd.DataFrame(feature_section_mapping),
                    feature_list=feature_list)

    # TODO: add tests, alterate file formats
    def read_file_data(self, filename: str, nlines: int = None, start_line: int = 0) -> dict:
        """
        Reads in un-annotated files expecting organization to be one document per line

        Sentence tokenizer will take care of splitting documents into sentences

        ex. parsed_data = read_file_data(filename='../tests/data/parse_and_model/oneLinePerDoc.txt', nlines=3)

        :param filename: Filename for the un-annotated data set
        :param nlines: Maximum number of lines from the file to read or None to read all lines
        :param start_line: Optional parameter, specific line number to start at, mostly for testing purposes
        :return: a dictionary with the following data
            section_list: DataFrame with the following form
                | doc_id (int)  | section_id (int)  | section_text (str)    | title (bool)  |
                doc_id: integer id for the document
                section_id: integer id for the section
                section_text: cleaned (lowercase, trimmed) section text
                title: True if the line is a title, False otherwise
        """
        logging.info("Reading data from file: ", filename)
        doc_id = -1
        section_id = 0
        section_list = []
        line_number = 0

        with open(filename, 'r') as input_file:
            for line in input_file:

                # Skip line if before specified start
                if line_number < start_line:
                    # Increment line number
                    line_number += 1
                    continue
                else:
                    # Increment line number
                    line_number += 1

                # Each line is new doc
                doc_id += 1

                # Parse doc and split into sentences
                # TODO: add a sentence tokenizer here
                sentence_list = sent_tokenize(line)
                for sentence in sentence_list:
                    logging.debug(sentence)

                    # Add section line to data set
                    section_list.append(
                        {"doc_id": doc_id, "section_id": section_id, "section_text": sentence})

                    # Increment section id
                    section_id += 1
                    logging.debug(line)

                # Check if max number of lines has been reached yet
                if nlines is not None:
                    if doc_id + 1 >= nlines:
                        break

        # Bundle and return data set
        return dict(section_list=pd.DataFrame(section_list))

    # TODO: Slow, needs to be optimized, unit tests need to be added
    def build_explicit_models(self, remove_stopwords: bool = True,
                              lemmatize_words: bool = True,
                              log_base: int = None) -> dict:
        """
        This function builds a background model, set of topic models and summarizes the counts of words in each sentence
            to prepare for EM optimization

        :param remove_stopwords: Set to true if stop words should be removed from document sections before models are
            created
        :param lemmatize_words: Set to true if lemmatization should be performed on document sections before models are
            created
        :param log_base: Optional parameter to specify log base, defaults to ln if not set
        :return: a dictionary with six entries -
            model_background: background model estimated from the entire document collection as described in section 4.2
            model_feature: feature models estimated from explicit mention sections as described in section 4.2
            model_background_matrix: background model estimated from the entire document collection as described in
                section 4.2 in dense ndarray form as follows:
                Word           | Background probability
                ----------------------------------------
                word 1         |   p(w1 | B)
                ...        ... |   ...
                word v         |   p(wv | B)
            model_feature_matrix: feature models estimated from explicit mention sections as described in section 4.2 in
                dense ndarray form as follows:
                Word/feature    | feature 1 ...     ...     feature k
                -----------------------------------------------------
                word 1         | p(w1 | f1) ...     ...     p(w1 | fk)
                ...        ...              ...            ...     ...
                word v         | p(wv | fk) ...     ...     p(wv | fk)
            section_word_counts_matrix: sparse csr matrix with word counts in each section as needed by the EM alg as
                follows:
                Section/Word | word 1 ... ... ... ... word v
                ---------------------------------------------------
                Section 1    | count(s_1,w_1) ... ...  count(s_1, w_v)
                Section 2    | count(s_2,w_2) ... ...  count(s_2, w_v)
                ...    ...     ... ...     ...     ...     ...
                Section m    | count(s_m, w_1)... ...  count(s_m, w_v)
            vocabulary_lookup: a dictionary with
                key: word id used in models, matrices, etc.
                value: actual word
        """
        logging.info("Building explicit models.")
        text_set = self.parsed_text["section_list"]
        feature_set = self.formatted_feature_list

        section_word_list = dict()  # list of all words in each section
        section_word_counts = dict()  # count of words in each section
        collection_word_counts = Counter()  # count of all words in all section
        word_section_counter = Counter()  # count of number of sections with word
        feature_word_counter = defaultdict(
            Counter)  # keep track of words appearing in section w/ explicit feature mention
        feature_section_mapping = []  # keeps a list of the sentence ids associated with each feature (many-to-many mapping)

        vocabulary = OrderedDict()
        current_word_id = -1

        unique_feature_ids = feature_set.feature_id.unique()

        # initialize Spacy model
        nlp = en_core_web_sm.load()

        annotated_text = text_set['section_text'].values.tolist()
        docs = nlp.pipe(annotated_text, batch_size=1000, n_threads=4)
        section_list = []
        for item in docs:
            section_list.append(item)

        # loop over all rows in input data set
        for index, row in text_set.iterrows():
            # print the current text for debugging
            logging.debug(str(row["section_id"]) + ": " + row["section_text"])

            # input the sentence into Spacy
            section = section_list[index]  # nlp(row["section_text"])

            # add each parsed word into a list
            current_section_words = []
            for word in section:
                if word.lower_.strip() == '':
                    continue

                # convert word to lowercase, strip stop words and lemmatize if requested
                if (not word.is_stop or not remove_stopwords) and not word.is_punct:
                    cleaned_word = word.lemma_ if (word.lemma_ != '-PRON-' and lemmatize_words) else word.lower_
                    current_section_words.append(cleaned_word)

                    # assign word an id if it doesn't have one already
                    if cleaned_word not in vocabulary:
                        current_word_id += 1
                        vocabulary[cleaned_word] = current_word_id

            # get a count of distinct words in the section - this might need to be switched to default dict later
            current_section_word_counts = Counter(current_section_words)

            # get keys for distinct words to add to idf counter
            word_section_counter.update(current_section_word_counts.keys())

            # add these counts to the all section counter
            collection_word_counts.update(current_section_words)

            # add to section counts dictionary
            section_word_counts[row["section_id"]] = current_section_word_counts

            # add to dictionary holding word parsing
            section_word_list[row["section_id"]] = current_section_words

            # initialize list to keep track of found features (in case of synonyms)
            found_features = set()

            # get all explicit topics for this sentence and add these words to the list
            for index_f, row_f in feature_set.iterrows():

                # word was found in the section, record find and add words to feature topic model
                if row_f["feature"] in current_section_words:
                    logging.debug("feature " + str(row_f["feature_id"]))

                    if row_f["feature_id"] in found_features:
                        # already found explicit feature mention in sentence as synonym, skip
                        continue
                    else:
                        # feature has not been found yet, add to the list
                        found_features.add(row_f["feature_id"])

                    # record that feature was explicitly found
                    feature_section_mapping.append({"section_id": row["section_id"], "feature_id": row_f["feature_id"]})

                    # if we only count each word once
                    # feature_word_counter[row_f["feature_id"]].update(current_section_word_counts.keys())

                    # if we count each words as many times as it occurs (consistent with Santu's code)
                    feature_word_counter[row_f["feature_id"]].update(current_section_words)

        # At this point we have all the counts we need to build the topic models

        ####################################
        # Calculations for background model
        ####################################

        # total number of words
        vocabulary_size = len(collection_word_counts.values())
        total_word_count = sum(collection_word_counts.values())

        # change counter to dictionary for calculations
        collection_word_counts = dict(collection_word_counts)

        # calculate background model - ensure words are in key order
        model_background = []
        for word, word_id in vocabulary.items():
            model_background.append(collection_word_counts[word] / total_word_count)

        ###############################
        # Calculations for topic model
        ###############################
        tfidf_feature = defaultdict(dict)
        model_feature_norms = Counter()

        # count of sentences
        section_count = len(section_word_list)

        for word in collection_word_counts.keys():

            for current_feature in unique_feature_ids:

                #######################################################################################################
                # Formula 4, section 4.2, using base e logs by default but can be changed, also adds +1 from Formula 5
                #######################################################################################################
                if log_base is None:
                    tfidf = math.log(1 + feature_word_counter[current_feature][word]) \
                            * math.log(1 + section_count / word_section_counter[word]) \
                            + 1
                else:
                    tfidf = math.log(1 + feature_word_counter[current_feature][word], log_base) \
                            * math.log(1 + section_count / word_section_counter[word], log_base) \
                            + 1
                logging.debug(str(tfidf))

                tfidf_feature[current_feature][word] = tfidf

                model_feature_norms[current_feature] += tfidf

        # normalize values of all dictionaries with totals
        model_feature = []

        for index in feature_set["feature_id"].unique():
            logging.debug("normalizing " + str(index))

            #########################################################################################################
            # Formula 5, section 4.2, using base e logs by default, +1 in numerator already taken care of in tfidf calculation
            #########################################################################################################
            model_feature.append([])
            for word, word_id in vocabulary.items():
                logging.debug(word + ":" + str(word_id))
                model_feature[index].append(tfidf_feature[index][word] / (model_feature_norms[index]))

        # translate section word counts into matrix for EM
        section_word_counts_matrix = np.zeros(shape=(section_count, vocabulary_size))
        for section, word_list in section_word_counts.items():
            logging.debug(section)
            for word, word_count in word_list.items():
                # look up current word
                word_id = vocabulary[word]
                logging.debug(str(word_id) + ":" + word)
                section_word_counts_matrix[section, word_id] = word_count

        # translate models into matrices for EM
        model_background_matrix = csr_matrix(np.array(model_background).T)
        model_feature_matrix = np.array(model_feature).T

        # reverse vocabulary dictionary so it can be used to back-translate later
        vocabulary_lookup = {v: k for k, v in vocabulary.items()}

        # Save model results to object
        return dict(model_background=model_background, model_feature=model_feature,
                    section_word_counts_matrix=csr_matrix(section_word_counts_matrix),
                    model_background_matrix=model_background_matrix, model_feature_matrix=model_feature_matrix,
                    vocabulary_lookup=vocabulary_lookup)

    # TODO: Try metapy impelementation (should mostly be a swap of the NLP call)
    def build_explicit_models_metapy(text_set: pd.DataFrame, feature_set: pd.DataFrame,
                                     remove_stopwords: bool = False,
                                     lemmatize_words: bool = True) -> dict:
        """
        This function builds a background model, set of topic models and summarizes the counts of words in each sentence
            to prepare for EM optimization

            ex. em_input = build_explicit_models(text_set=text_set, feature_set=feature_set)
            ex. em_input = build_explicit_models(text_set = annotated_data["section_list"], feature_set = feature_list)

        :param text_set: a pandas DataFrame with (at a minimum) the following columns
        :param feature_set: output of format feature list - OR -
            DataFrame with integer ids for each feature, synonyms are grouped together
            | feature (str) | feature_id (int)  | feature_term_id (int)
            feature: string representation of the feature
            feature_id: integer id for the feature, will be the same for synonyms if input in nested list
            feature_term_id: integer id for the feature, will be unique for each string, including synonyms
        :param lemmatize_words: set to true if lemmatization should be performed on document sections before models are
            created
        :param remove_stopwords: set to true if stop words should be removed from document sections before models are
            created
        :return: a dictionary with seven entries -
            model_background: background model estimated from the entire document collection as described in section 4.2
            model_background_matrix: background model estimated from the entire document collection as described in
                section 4.2 in array form as follows:
                Word           | Background probability
                ----------------------------------------
                word 1         |   bp_1
                ...        ... |   ...
                word |V|        |   bp_|V|
            model_feature: feature models estimated from explicit mention sections as described in section 4.2
            model_feature_matrix: feature models estimated from explicit mention sections as described in section 4.2 in
                array form as follows:
                Word/Feature    | Feature 1   ...     ...     Feature |f|=k
                -----------------------------------------------------------
                word 1         | p(w1 | f1) ...      ...    p(w1 | fk)
                ...        ...             ....            ...     ...
                word nw        | p(w_|V|, fk) ... ....   tm(w_|V|, fk)
            section_word_counts: word counts in each section as needed by the EM algorithm
            section_word_counts_matrix: word counts in each section in matrix form as needed by the EM alg as follows:
                Section/Word | word 1 ... ... ... ... word |V|
                ----------------------------------------------------------
                Section 1    | count(s_1,w_1) ... ...  count(s_1, w_|V|)
                Section 2    | count(s_2,w_1) ... ...  count(s_2, w_|V|)
                ...    ...     ... ...     ...     ...     ...
                Section m    | count(s_m, w_1)... ...  count(s_m, w_|V|)
            vocabulary_lookup: a dictionary with
                key: word id used in models, matrices, etc.
                value: actual word
        """
        return ()

    # TODO: Try nltk impelementation (should mostly be a swap of the NLP call)
    def build_explicit_models_nltk(text_set: pd.DataFrame, feature_set: pd.DataFrame,
                                   remove_stopwords: bool = False,
                                   lemmatize_words: bool = True) -> dict:
        """
        This function builds a background model, set of topic models and summarizes the counts of words in each sentence
            to prepare for EM optimization

            ex. em_input = build_explicit_models(text_set=text_set, feature_set=feature_set)
            ex. em_input = build_explicit_models(text_set = annotated_data["section_list"], feature_set = feature_list)

        :param text_set: a pandas DataFrame with (at a minimum) the following columns
        :param feature_set: output of format feature list - OR -
            DataFrame with integer ids for each feature, synonyms are grouped together
            | feature (str) | feature_id (int)  | feature_term_id (int)
            feature: string representation of the feature
            feature_id: integer id for the feature, will be the same for synonyms if input in nested list
            feature_term_id: integer id for the feature, will be unique for each string, including synonyms
        :param lemmatize_words: set to true if lemmatization should be performed on document sections before models are
            created
        :param remove_stopwords: set to true if stop words should be removed from document sections before models are
            created
        :return: a dictionary with seven entries -
            model_background: background model estimated from the entire document collection as described in section 4.2
            model_background_matrix: background model estimated from the entire document collection as described in
                section 4.2 in array form as follows:
                Word           | Background probability
                ----------------------------------------
                word 1         |   bp_1
                ...        ... |   ...
                word |V|        |   bp_|V|
            model_feature: feature models estimated from explicit mention sections as described in section 4.2
            model_feature_matrix: feature models estimated from explicit mention sections as described in section 4.2 in
                array form as follows:
                Word/Feature    | Feature 1   ...     ...     Feature |f|=k
                -----------------------------------------------------------
                word 1         | p(w1 | f1) ...      ...    p(w1 | fk)
                ...        ...             ....            ...     ...
                word nw        | p(w_|V|, fk) ... ....   tm(w_|V|, fk)
            section_word_counts: word counts in each section as needed by the EM algorithm
            section_word_counts_matrix: word counts in each section in matrix form as needed by the EM alg as follows:
                Section/Word | word 1 ... ... ... ... word |V|
                ----------------------------------------------------------
                Section 1    | count(s_1,w_1) ... ...  count(s_1, w_|V|)
                Section 2    | count(s_2,w_1) ... ...  count(s_2, w_|V|)
                ...    ...     ... ...     ...     ...     ...
                Section m    | count(s_m, w_1)... ...  count(s_m, w_|V|)
            vocabulary_lookup: a dictionary with
                key: word id used in models, matrices, etc.
                value: actual word
        """
        return ()


if __name__ == '__main__':
    print("CWD:", os.getcwd())
    start_time = time.time()

    print("Calling ParseAndModel...")
    pm = ParseAndModel(feature_list=["sound", "battery", ["screen", "display"]],
                       filename='../tests/data/parse_and_model/iPod.final')

    print(pm.model_results)

    end_time = time.time()
    print("Elapsed: {} seconds".format(round(end_time - start_time, 4)))
