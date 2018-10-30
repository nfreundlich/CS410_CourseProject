import pandas as pd
import spacy
import en_core_web_sm

text_set = pd.read_csv('demo_files/sample_dataset_1_text.csv')

feature_set = pd.read_csv('demo_files/sample_dataset_1_features.csv')

feature_mapping = pd.read_csv('demo_files/sample_dataset_1_feature_mapping.csv')

print(text_set.head())
print(feature_set.head())
print(feature_mapping.head())

nlp = spacy.load('en_core_web_sm')
