#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import spacy
import timeit
from IPython.display import HTML, IFrame
from w3lib.html import remove_tags
from tqdm import tqdm


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
path = 'aclImdb/'

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
path = 'aclImdb/'
# dataset = pd.read_csv(os.path.join(path, 'dataset_raw/dataset_raw.csv'))
dataset = pd.read_csv(os.path.join(path, 'dataset_raw.csv'))

# Define batch_size and n_threads
batch_size = 512
n_threads = 2

dataset_raw = pd.read_csv('aclImdb/dataset_raw.csv')

# Print the first 5 rows from the dataset
dataset_raw.head()
dataset_raw.info()

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

# Test the processing time on a part of the trainig set, given batch_size and n_threads
print('Start processing 1000 examples using batch_size: {} and n_threads: {}'.format(batch_size, n_threads))
start_time = timeit.default_timer()
text_preprocessing(dataset_raw.loc[:1000, 'review'], batch_size=batch_size, n_threads=n_threads)
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
        for frame in tqdm(data_frames):
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
path = os.path.join(os.getcwd(), 'aclImdb/')
name = 'dataset_feat_clean.csv'

# Perform text preprocessing and save the resulted frame to CSV file
split_norm_save(dataset_raw, name, path, part_size, batch_size, n_threads, nlp)

# Import preprocessed dataset from CSV file
# dataset_feat_clean = pd.read_csv('dataset/datasets_feat_clean/dataset_feat_clean.csv')
dataset_feat_clean = pd.read_csv('aclImdb/dataset_feat_clean.csv')
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
train_set.to_csv('aclImdb/train_feat_clean.csv', index=False)
val_set.to_csv('aclImdb/val_feat_clean.csv', index=False)
test_set.to_csv('aclImdb/test_feat_clean.csv', index=False)
