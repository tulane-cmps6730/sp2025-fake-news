# -*- coding: utf-8 -*-

"""Demonstrating a very simple NLP project. Yours should be more exciting than this."""
import click
import glob
import pickle
import sys
import os

import numpy as np
import pandas as pd
import re
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# Kaggle API is only needed for the `dl-data` command; avoid import at module
# load time so other commands (e.g. stance) work without a kaggle.json file.

from . import clf_path, config
# NOTE: preprocessing utilities require SpaCy model; import lazily in the command

def project_data_dir():
    # <repo-root>/nlp/data   (independent of CWD or cfg)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

@click.group()
def main(args=None):
    """Console script for nlp."""
    return 0

@main.command('web')
@click.option('-p', '--port', required=False, default=5000, show_default=True, help='port of web server')
def web(port):
    """
    Launch the flask web app.
    """
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)
    
@main.command('dl-data')
def dl_data():
    """
    Download training/testing data from Kaggle.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi  # late import
    # Initialize the API
    api = KaggleApi()
    api.authenticate()
    
    # List of datasets to download
    datasets = [
        'emineyetm/fake-news-detection-datasets',          # Fake/True news
        'subodh7300/gossipcop',                            # GossipCop
        'khandalaryan/liar-preprocessed-dataset',          # LIAR dataset
    ]

    print("Datasets to download:")
    for ds in datasets:
        print(f"  • {ds}")
    
    # Hard-code the data directory to be nlp/data, resolved as absolute path
    data_dir = project_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    print(f'Data will be downloaded to: {data_dir}')
    
    # Download each dataset
    for dataset in datasets:
        print(f"\nDownloading {dataset} …")
        api.dataset_download_files(
            dataset,
            path=data_dir,
            unzip=True,
            quiet=False
        )
        print(f"Finished {dataset}")

    print(f"\nAll datasets downloaded to {data_dir}")

def data2df():
    """
    Read the dataset files from the data directory.
    """
    data_dir = project_data_dir()
    fake_df = pd.read_csv(os.path.join(data_dir, 'Fake.csv'))
    true_df = pd.read_csv(os.path.join(data_dir, 'True.csv'))
    return fake_df, true_df

@main.command('stats')
def stats():
    """
    Read the data files and print interesting statistics.
    """
    fake_df, true_df = data2df()
    print('Fake news dataset:')
    print(f'{len(fake_df)} rows')
    print('\nTrue news dataset:')
    print(f'{len(true_df)} rows')

@main.command('train')
def train():
    """
    Train a classifier and save it.
    """
    # (1) Read the data...
    df = data2df()    
    # (2) Create classifier and vectorizer.
    clf = LogisticRegression(max_iter=1000, C=1, class_weight='balanced')         
    vec = CountVectorizer(min_df=5, ngram_range=(1,3), binary=True, stop_words='english')
    X = vec.fit_transform(df.title)
    y = df.partisan.values
    # (3) do cross-validation and print out validation metrics
    # (classification_report)
    do_cross_validation(clf, X, y)
    # (4) Finally, train on ALL data one final time and
    # train. Save the classifier to disk.
    clf.fit(X, y)
    pickle.dump((clf, vec), open(clf_path, 'wb'))
    top_coef(clf, vec)

def do_cross_validation(clf, X, y):
    all_preds = np.zeros(len(y))
    for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X,y):
        clf.fit(X[train], y[train])
        all_preds[test] = clf.predict(X[test])
    print(classification_report(y, all_preds))    

def top_coef(clf, vec, labels=['liberal', 'conservative'], n=10):
    feats = np.array(vec.get_feature_names_out())
    print('top coef for %s' % labels[1])
    for i in np.argsort(clf.coef_[0])[::-1][:n]:
        print('%20s\t%.2f' % (feats[i], clf.coef_[0][i]))
    print('\n\ntop coef for %s' % labels[0])
    for i in np.argsort(clf.coef_[0])[:n]:
        print('%20s\t%.2f' % (feats[i], clf.coef_[0][i]))

@main.command('preprocess')
def preprocess():
    """
    Preprocess the data and split into train/test sets.
    """
    # Import preprocessing utilities lazily to avoid heavy deps unless needed
    from .helper_script.preprocessing import load_data, prepare_dataset
    
    # Get data directory from config
    data_dir = project_data_dir()
    
    # Define paths to the datasets
    fake_path = os.path.join(data_dir, 'News _dataset', 'Fake.csv')
    true_path = os.path.join(data_dir, 'News _dataset', 'True.csv')
    
    print("Loading datasets...")
    df = load_data(fake_path, true_path)
    print(f"Total samples: {len(df)}")
    
    print("\nPreparing train/test split and preprocessing text...")
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    
    print("\nDataset split complete:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Save preprocessed data
    preprocessed_dir = os.path.join(data_dir, 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Save train and test sets
    train_data = pd.concat([X_train, pd.Series(y_train, name='label')], axis=1)
    test_data = pd.concat([X_test, pd.Series(y_test, name='label')], axis=1)
    
    train_data.to_csv(os.path.join(preprocessed_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(preprocessed_dir, 'test.csv'), index=False)
    print(f"\nPreprocessed data saved to {preprocessed_dir}")

# --------------------
# Stance detection CLI
# --------------------

@main.command('stance')
@click.option('--infile', type=click.Path(exists=True), required=True,
              help='CSV with text & claim columns')
@click.option('--outfile', type=click.Path(), required=True,
              help='Where to save CSV with added stance column')
@click.option('--model', default='roberta-large-mnli', show_default=True,
              help='HF model name for zero-shot classification')
@click.option('--sample', type=int, default=None,
              help='Randomly sample N rows before prediction for speed')
def stance_cmd(infile, outfile, model, sample):
    """Run zero-shot stance detection and save results."""
    from .stance_detection import load_classifier, batch_predict
    click.echo(f'Loading data from {infile} …')
    df = pd.read_csv(infile)
    if sample is not None and sample < len(df):
        click.echo(f'Sampling {sample} rows from {len(df)} …')
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
    clf = load_classifier(model)
    click.echo('Predicting stances …')
    df = batch_predict(df, clf)
    os.makedirs(os.path.dirname(outfile) or '.', exist_ok=True)
    df.to_csv(outfile, index=False)
    click.echo(f'Saved predictions to {outfile}')

@main.command('data-dir')
def show_data_dir():
    """Print the absolute path where data files are stored (nlp/data)."""
    click.echo(project_data_dir())

if __name__ == "__main__":
    sys.exit(main())
