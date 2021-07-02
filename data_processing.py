#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import scattertext as st
import collections
from IPython.display import HTML, IFrame
from textblob import TextBlob
from w3lib.html import remove_tags
from wordcloud import WordCloud
from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA


def load_data(path, file_list, dataset, encoding='utf8'):
    """Read set of files from given directory and save returned lines to list.

    Parameters
    ----------
    path : str
        Absolute or relative path to given file (or set of files).
    file_list: list
        List of files names to read.
    dataset: list
        List that stores read lines.
    encoding: str, optional (default='utf8')
        File encoding.

    """
    for file in file_list:
        with open(os.path.join(path, file), 'r', encoding=encoding) as text:
            dataset.append(text.read())

# path = 'dataset/'
path = 'aclImdb_v1/'

train_pos, train_neg, test_pos, test_neg = [], [], [], []

# sets_dict = {'train/positive/': train_pos, 'train/negative/': train_neg,
#              'test/positive/': test_pos, 'test/negative/': test_neg}
sets_dict = {'train/pos/': train_pos, 'train/neg/': train_neg,
             'test/pos/': test_pos, 'test/neg/': test_neg}

for dataset in sets_dict:
    file_list = [f for f in os.listdir(os.path.join(path, dataset)) if f.endswith('.txt')]
    load_data(os.path.join(path, dataset), file_list, sets_dict[dataset])

dataset = pd.concat([pd.DataFrame({'review': train_pos, 'label': 1}),
                     pd.DataFrame({'review': test_pos, 'label': 1}),
                     pd.DataFrame({'review': train_neg, 'label': 0}),
                     pd.DataFrame({'review': test_neg, 'label': 0})],
                    axis=0, ignore_index=True)

dataset.head()
dataset.tail()

dataset.label.value_counts()
dataset.info()
dataset.isna().sum()

duplicate_indices = dataset.loc[dataset.duplicated(keep='first')].index
print('Number of duplicates in the dataset: {}'.format(dataset.loc[duplicate_indices, 'review'].count()))

dataset.loc[duplicate_indices, :].head()
dataset.drop_duplicates(keep='first', inplace=True)
print('Dataset shape after removing duplicates: {}'.format(dataset.shape))

HTML(dataset.iloc[np.random.randint(dataset.shape[0]), 0])
# dataset.to_csv(os.path.join(path, 'dataset_raw/dataset_raw.csv'), index=False)
dataset.to_csv(os.path.join(path, 'dataset_raw.csv'), index=False)

# path = 'dataset/'
path = 'aclImdb_v1/'
# dataset = pd.read_csv(os.path.join(path, 'dataset_raw/dataset_raw.csv'))
dataset = pd.read_csv(os.path.join(path, 'dataset_raw.csv'))

def polarity(text):
    """Calculate the polarity score of the input text.

    """
    return TextBlob(text).sentiment.polarity

def subjectivity(text):
    """Calculate the subjectivity score of the input text.

    """
    return TextBlob(text).sentiment.subjectivity

def pos(df, batch_size, n_threads, required_tags):
    """Count the number of peculiar POS tags in data series of strings.

    Parameters
    ----------
    df : pandas.Series
        Pandas.Series containing strings to process.
    batch_size: int
        Size of text batch (recommended to be the power of 2).
    n_threads: int
        Number of threads in multiprocessing.
    required_tags: list
        List containing spacy's POS tags to count.

    Returns
    -------
    pandas.DataFrame
       DataFrame of a shape (index, len(required_tags)).

    """
    # Add index column to reviews frame and change column order
    reviews = df.reset_index(drop=False)[['review', 'index']]
    # Convert dataframe to list of tuples (review, index)
    review_list = list(zip(*[reviews[c].values.tolist() for c in reviews]))
    # Create empty dictionary
    review_dict = collections.defaultdict(dict)

    for doc, context in list(nlp.pipe(review_list, as_tuples=True, batch_size=batch_size, n_threads=n_threads)):
        review_dict[context] = {}
        for token in doc:
            pos = token.pos_
            if pos in required_tags:
                review_dict[context].setdefault(pos, 0)
                review_dict[context][pos] = review_dict[context][pos] + 1
    # Transpose data frame to shape (index, tags)
    return pd.DataFrame(review_dict).transpose()

def pos2(df, batch_size, n_threads, required_tags):
    """Count the number of peculiar POS tags in data series of strings.

    Parameters
    ----------
    df : pandas.Series
        Pandas.Series containing strings to process.
    batch_size: int
        Size of text batch (recommended to be the power of 2).
    n_threads: int
        Number of threads in multiprocessing.
    required_tags: list
        List containing spacy's POS tags to count.

    Returns
    -------
    pandas.DataFrame
       DataFrame of a shape (index, len(required_tags)).

    """
    # Create empty dictionary
    review_dict = collections.defaultdict(dict)
    for i, doc in enumerate(nlp.pipe(df, batch_size=batch_size, n_threads=n_threads)):
        for token in doc:
            pos = token.pos_
            if pos in required_tags:
                review_dict[i].setdefault(pos, 0)
                review_dict[i][pos] = review_dict[i][pos] + 1
    # Transpose data frame to shape (index, tags)
    return pd.DataFrame(review_dict).transpose()

def pos3(df, required_tags):
    """Count the number of peculiar POS tags in data series of strings.

    Parameters
    ----------
    df : pandas.Series
        Pandas.Series containing strings to process.
    required_tags: list
        List containing spacy's POS tags to count.

    Returns
    -------
    pandas.DataFrame
       DataFrame of a shape (index, len(required_tags)).

    """
    pos_list = []
    for i in range(df.shape[0]):
        doc = nlp(df[i])
        pos_dict = {}
        for token in doc:
            pos = token.pos_
            if pos in required_tags:
                pos_dict.setdefault(pos, 0)
                pos_dict[pos] = pos_dict[pos] + 1
        pos_list.append(pos_dict)
    return pd.DataFrame(pos_list)

# Load language model and disable unnecessary components of processing pipeline
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat', '...'])
required_tags = ['PROPN', 'PUNCT', 'NOUN', 'ADJ', 'VERB']

# Define batch_size and n_threads
batch_size = 512
n_threads = 2

# Test the processing time on a part of the dataset, given batch_size and n_threads
start_time = timeit.default_timer()
print('Start processing 1000 examples using batch_size: {} and n_threads: {}'.format(batch_size, n_threads))
pos(dataset.loc[:1000, 'review'], required_tags=required_tags, batch_size=batch_size, n_threads=n_threads)
print('Function 1 processing time: {:.2f} sec'.format(timeit.default_timer() - start_time))

# Define batch_size and n_threads
batch_size = 512
n_threads = 2

# Test the processing time on a part of the dataset, given batch_size and n_threads
start_time = timeit.default_timer()
print('Start processing 1000 examples using batch_size: {} and n_threads: {}'.format(batch_size, n_threads))
pos2(dataset.loc[:1000, 'review'], required_tags=required_tags, batch_size=batch_size, n_threads=n_threads)
print('Function 2 processing time: {:.2f} sec'.format(timeit.default_timer() - start_time))

# Test the processing time on a part of the dataset, given batch_size and n_threads
start_time = timeit.default_timer()
print('Start processing 1000 examples')
pos3(dataset.loc[:1000, 'review'], required_tags=required_tags)
print('Function 3 processing time: {:.2f} sec'.format(timeit.default_timer() - start_time))

def extract_features(df, batch_size, n_threads, required_tags):
    """Extract the following features from the data frame's 'review' column:
    polarity, subjectivity, word_count, UPPERCASE, DIGITS, and POS tags specified by required_tags.

    Convert extracted features to int16 or float16 data types.

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas.DataFrame containing 'review' column to which extraction will be applied.
    batch_size: int
        Size of text batch (recommended to be the power of 2).
    n_threads: int
        Number of threads in multiprocessing.
    required_tags: list
        List containing spacy's POS tags to count.

    Returns
    -------
    pandas.DataFrame
       Concatenation of the original data frame and data frame containing extracted features.

    """
    # Calculate polarity
    df['polarity'] = df.review.apply(polarity).astype('float16')
    # Calculate subjectivity
    df['subjectivity'] = df.review.apply(subjectivity).astype('float16')
    # Calculate number of words in review
    df['word_count'] = df.review.apply(lambda text: len(text.split())).astype('int16')
    # Count number of uppercase words, then divide by word_count
    df['UPPERCASE'] = df.review.apply(
        lambda text: len([word for word in text.split() if word.isupper()])) / df.word_count
    # Change data type to float16
    df.UPPERCASE = df.UPPERCASE.astype('float16')
    # Count number of digits, then divide by word_count
    df['DIGITS'] = df.review.apply(lambda text: len([word for word in text.split() if word.isdigit()])) / df.word_count
    # Change data type to float16
    df.DIGITS = df.DIGITS.astype('float16')
    # Perform part-of-speech taging
    pos_data = pos2(df.review, batch_size=batch_size, n_threads=n_threads, required_tags=required_tags)
    # Divide POS tags count by word_count
    pos_data = pos_data.div(df.word_count, axis=0).astype('float16')
    # Concatenate pandas data frames horizontaly
    return pd.concat([df, pos_data], axis=1)

# Load language model and disable unnecessary components of processing pipeline
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat', '...'])
required_tags = ['PROPN', 'PUNCT', 'NOUN', 'ADJ', 'VERB']

batch_size = 512
n_threads = 2

# Test the processing time on a part of the trainig set, given batch_size and n_threads
start_time = timeit.default_timer()
print('Start processing 1000 examples using batch_size: {} and n_threads: {}'.format(batch_size, n_threads))
extract_features(dataset.loc[:1000, :], batch_size=batch_size, n_threads=n_threads, required_tags=required_tags)
print('Feature extraction function processing time: {:.2f} sec'.format(timeit.default_timer() - start_time))

def split_extract_save(df, name, path, part_size, batch_size, n_threads, required_tags, nlp):
    """Split data frame into chunks of size equal: part_size and perform feature extraction on each of the parts.
    Extract the following features from the data frame part's 'review' column: polarity, subjectivity, word_count,
    UPPERCASE, DIGITS, and POS tags specified by required_tags.

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas.DataFrame containing 'review' column to which extraction will be applied.
    name : str
        Name of the CSV file to which export the data.
    path : str
        Absolute or relative path to directory where to save the data.
    part_size: int
        Size of the chunk to process (number of strings it contains).
    batch_size: int
        Size of text batch (recommended to be the power of 2).
    n_threads: int
        Number of threads in multiprocessing.
    required_tags: list
        List containing spacy's POS tags to count.
    nlp: spacy.lang.<language>
        Spacy language model (for example spacy.lang.en.English)

    Returns
    -------
    pandas.DataFrame
       Concatenation of the original data frame and data frame containing extracted features.

    """
    if name not in os.listdir(path):
        dataset_parts = []
        N = int(len(df) / part_size)
        # Create list of dataframe chunks
        data_frames = [df.iloc[i * part_size:(i + 1) * part_size].copy() for i in range(N + 1)]
        # Process dataset partialy
        for frame in tqdm_notebook(data_frames):
            # Extract features from dataset chunk
            dataset_part = extract_features(frame, batch_size=batch_size, n_threads=n_threads,
                                            required_tags=required_tags)
            dataset_parts.append(dataset_part)
            # Reload nlp
            nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat', '...'])

        # Concatenate all parts into one dataset
        dataset_feat = pd.concat(dataset_parts, axis=0, sort=False)
        # Replace missing values NaN with 0
        dataset_feat.fillna(0, inplace=True)
        # Convert label values to int16
        dataset_feat.label = dataset_feat.label.astype('int16')
        # Export data frame to CSV file
        dataset_feat.to_csv(path + name, index=False)
    else:
        print('File {} already exists in given directory.'.format(name))

# Define all required variables
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat', '...'])
required_tags = ['PROPN', 'PUNCT', 'NOUN', 'ADJ', 'VERB']
batch_size = 512
n_threads = 2
part_size = 5000
# path = os.path.join(os.getcwd(), 'dataset/datasets_feat/')
path = os.path.join(os.getcwd(), 'aclImdb_v1/')
name = 'dataset_feat.csv'

# Perform feature extraction and export resulted file into CSV
split_extract_save(dataset, name, path, part_size, batch_size, n_threads, required_tags, nlp)

# Dictionary of {column: dtype} pairs
col_types = {'review': str, 'label': np.int16, 'polarity': np.float16, 'subjectivity': np.float16,
             'word_count': np.int16, 'UPPERCASE': np.float16, 'DIGITS': np.float16, 'PROPN': np.float16,
             'VERB': np.float16, 'NOUN': np.float16, 'PUNCT': np.float16, 'ADJ': np.float16}

# Import dataset from the CSV file
# dataset_feat = pd.read_csv('dataset/datasets_feat/dataset_feat.csv', dtype=col_types)
dataset_feat = pd.read_csv('aclImdb_v1/dataset_feat.csv', dtype=col_types)

# Print the first 5 rows from the dataset
dataset_feat.head()
dataset_feat.info()
# Import the dataset
# dataset_feat = pd.read_csv('dataset/datasets_feat/dataset_feat.csv')
dataset_feat = pd.read_csv('aclImdb_v1/dataset_feat.csv')

def token_filter(token):
    """Filter the token for text_preprocessing function.
    Check if the token is not: punctuation, whitespace, stopword or digit.

    Parameters
    ----------
    token : spacy.Token
        Token passed from text_preprocessing function.

    Returns
    -------
    Bool
       True if token meets the criteria, otherwise False.

    """
    return not (token.is_punct | token.is_space | token.is_stop | token.is_digit | token.like_num)


def text_preprocessing(df, batch_size, n_threads):
    """Perform text preprocessing using the following methods: removing HTML tags, lowercasing,
    lemmatization and removing stopwords, whitespaces, punctuations, digits.

    Parameters
    ----------
    df : pandas.Series
        Pandas.Series containing strings to process.
    batch_size: int
        Size of text batch (recommended to be the power of 2).
    n_threads: int
        Number of threads in multiprocessing.

    Returns
    -------
    pandas.Series
       Pandas.Series containing processed strings.

    """
    # Remove HTML tags
    df = df.apply(remove_tags)
    # Make lowercase
    df = df.str.lower()
    processed_docs = []
    for doc in list(nlp.pipe(df, batch_size=batch_size, n_threads=n_threads)):
        # Remove stopwords, spaces, punctutations and digits
        text = [token for token in doc if token_filter(token)]
        # Lemmatization
        text = [token.lemma_ for token in text if token.lemma_ != '-PRON-']
        processed_docs.append(' '.join(text))
    return pd.Series(processed_docs, name='clean_review', index=df.index)

# Define the variables
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat', 'tagger', '...'])
batch_size = 512
n_threads = 2

# Test the processing time on a part of the trainig set, given batch_size and n_threads
print('Start processing 1000 examples using batch_size: {} and n_threads: {}'.format(batch_size, n_threads))
start_time = timeit.default_timer()
text_preprocessing(dataset_feat.loc[:1000, 'review'], batch_size=batch_size, n_threads=n_threads)
print('Processing time: {:.2f} sec'.format(timeit.default_timer() - start_time))

def split_norm_save(df, name, path, part_size, batch_size, n_threads, nlp):
    """Split data frame into chunks of size equal: part_size and perform text preprocessing on each of the parts.
    Preprocess strings using the following methods: removing HTML tags, lowercasing, lemmatization and
    removing stopwords, whitespaces, punctuations, digits.

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas.DataFrame containing 'review' column to preprocess.
    name : str
        Name of the CSV file to which export the data.
    path : str
        Absolute or relative path to directory where to save the data.
    part_size: int
        Size of the chunk to process (number of strings it contains).
    batch_size: int
        Size of text batch (recommended to be the power of 2).
    n_threads: int
        Number of threads in multiprocessing.
    nlp: spacy.lang.<language>
        Spacy language model (for example spacy.lang.en.English)

    Returns
    -------
    pandas.DataFrame
       Concatenation of the original data frame and pandas series of normalized strings.

    """
    if name not in os.listdir(path):
        dataset_parts = []
        N = int(len(df) / part_size)
        # Create list of dataframe chunks
        data_frames = [df.iloc[i * part_size:(i + 1) * part_size, 0].copy() for i in range(N + 1)]
        # Process dataset partialy
        for frame in tqdm_notebook(data_frames):
            # Normalize dataset chunk
            dataset_part = text_preprocessing(frame, batch_size=batch_size, n_threads=n_threads)
            dataset_parts.append(dataset_part)
            # Reload nlp
            nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat', '...'])

        # Concatenate all parts into one series
        concat_clean = pd.concat(dataset_parts, axis=0, sort=False)
        # Concatenate dataset and cleaned review seires
        dataset_clean = pd.concat([df, concat_clean], axis=1)
        # Export data frame to CSV file
        dataset_clean.to_csv(path + name, index=False)
    else:
        print('File {} already exists in given directory.'.format(name))

# Define variables
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat', 'tagger', '...'])
batch_size = 512
n_threads = 2
part_size = 5000
# path = os.path.join(os.getcwd(), 'dataset/datasets_feat_clean/')
path = os.path.join(os.getcwd(), 'aclImdb_v1/')
name = 'dataset_feat_clean.csv'

# Perform text preprocessing and save the resulted frame to CSV file
split_norm_save(dataset_feat, name, path, part_size, batch_size, n_threads, nlp)

# Import preprocessed dataset from CSV file
# dataset_feat_clean = pd.read_csv('dataset/datasets_feat_clean/dataset_feat_clean.csv')
dataset_feat_clean = pd.read_csv('aclImdb_v1/dataset_feat_clean.csv')
dataset_feat_clean.head()


def train_val_test_split(df, val_size, test_size, random_state=0):
    """Split data frame into 3 (train/val/test) sets or into 2 (train/val) sets.

    If you want to split into two datasets, set test_size = 0.

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas.DataFrame to split.
    val_size : float
        Fraction of dataset to include in validation set. Should be from range (0.0, 1.0).
    test_size : float
        Fraction of dataset to include in test set. Should be from range <0.0, 1.0).
    random_state: int, optional (default=0)
        The seed used by the random number generator.

    Returns
    -------
    train: pandas.DataFrame
       Training set.
    val: pandas.DataFrame
       Validation set.
    test: pandas.DataFrame
       Test set.

    Raises
    ------
    AssertionError
        If the val_size and test_size sum is greater or equal 1 or the negative value was passed.

    """
    assert (val_size + test_size) < 1, 'Validation size and test size sum is greater or equal 1'
    assert val_size >= 0 and test_size >= 0, 'Negative size is not accepted'
    train, val, test = np.split(df.sample(frac=1, random_state=random_state),
                                [int((1 - (val_size + test_size)) * len(df)), int((1 - test_size) * len(df))])
    return train, val, test


train_set, val_set, test_set = train_val_test_split(dataset_feat_clean, val_size=0.20, test_size=0.10)

print('Training set shape: {}'.format(train_set.shape))
print('Validation set shape: {}'.format(val_set.shape))
print('Test set shape: {}'.format(test_set.shape))

# train_set.to_csv('dataset/datasets_feat_clean/train_feat_clean.csv', index=False)
# val_set.to_csv('dataset/datasets_feat_clean/val_feat_clean.csv', index=False)
# test_set.to_csv('dataset/datasets_feat_clean/test_feat_clean.csv', index=False)
train_set.to_csv('aclImdb_v1/train_feat_clean.csv', index=False)
val_set.to_csv('aclImdb_v1/val_feat_clean.csv', index=False)
test_set.to_csv('aclImdb_v1/test_feat_clean.csv', index=False)
