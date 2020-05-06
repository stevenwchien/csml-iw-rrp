from transformers import BertForSequenceClassification, BertTokenizer
import torch
import numpy as np
import pandas as pd
from torch.utils.data import SequentialSampler, DataLoader
from logger import Logger

from train_model import evaluate
from data_util import extract_features

import argparse
import time
import sys
import json

# helper method to create a confusion matrix using pandas crosstab method
def confusion_matrix(preds, labels):
    pred_labels = preds.astype(int) + 1
    actual_labels = labels.astype(int) + 1
    both = np.stack((pred_labels, actual_labels), axis=-1)
    labels_df = pd.DataFrame(data=both, columns=['predicted', 'actual'])

    return pd.crosstab(labels_df['predicted'], labels_df['actual'], rownames=['Predicted'], colnames=['Actual'])

# helper method to print the mean average distance
def mean_abs_dist(preds, labels):
    return np.sum(np.abs(preds-labels)) / len(labels)

def main():
    parser = argparse.ArgumentParser(description='argument parsing for testing')

    parser.add_argument('--data_dir',
    default='data',
    type=str,
    help='path to data directory - default: \'data\'')

    parser.add_argument('--review_file',
    default='yelp_reviews_test1000.csv',
    type=str,
    help='file name containig reviews')

    parser.add_argument('--batch_size',
    default=32,
    type=int,
    help='batch size - default: 32')

    parser.add_argument('--model_save',
    default='./model_save/',
    type=str,
    help='directory to pull model')

    parser.add_argument('--nolog',
    action='store_true',
    help='disable logging')

    # parse input arguments
    clargs = parser.parse_args()

    # log to file and stdout
    if clargs.nolog:
        print("Not logging")
    else:
        sys.stdout = Logger('test')

    print("")
    print("==========================================")
    print("-------------Confirm Arguments------------")
    print("==========================================")

    print("Data directory for test data: {0:s}".format(clargs.data_dir))
    print("Test reviews file: {0:s}".format(clargs.review_file))
    print("Batch size of {0:d}".format(clargs.batch_size))
    print("Loading model from: {0:s}".format(clargs.model_save))

    print("")
    print("==========================================")
    print("---------------Generate Data--------------")
    print("==========================================")

    path = clargs.data_dir
    fn = clargs.review_file
    filename = path + "/" + fn

    t0 = time.perf_counter()
    print("Reading in training data from {0:s}".format(clargs.review_file))
    reviews_df = pd.read_csv(filename)
    reviews_df = reviews_df[['text', 'stars']]
    TEST_SIZE = len(reviews_df.index)
    elapsed = time.perf_counter() - t0
    print("Finished reading {0:d} entries | Took {1:0.2f} seconds".format(TEST_SIZE, elapsed))

    # load the model from save
    print("")
    print("==========================================")
    print("----------------Load Model----------------")
    print("==========================================")

    print("Loading model and tokenizer from directory")
    model_path = clargs.model_save
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print("Tokenizing the data to be tested")
    dataset = extract_features(reviews_df, tokenizer)
    test_dataloader = DataLoader(
    dataset,
    sampler = SequentialSampler(dataset),
    batch_size = clargs.batch_size,
    drop_last = False)

    # load hyperparameters of the model
    json_infile = model_path + '/' + 'hyperparams.json'
    with open(json_infile, 'r') as infile:
        hyper_json = json.load(infile)

    # test the model against some test data
    print("")
    print("==========================================")
    print("----------------Test Model----------------")
    print("==========================================")
    print("Testing - Split {0:d} examples into {1:d} batches".format(TEST_SIZE, len(test_dataloader)))
    test_loss, test_acc, pred_labels, actual_labels = evaluate(model, device, test_dataloader, TEST_SIZE)
    mad = mean_abs_dist(pred_labels, actual_labels)
    conf_matrix = confusion_matrix(pred_labels, actual_labels)
    print("")
    print("==========================================")
    print("---------------TEST RESULTS---------------")
    print("==========================================")
    print("")
    print("-----------TRAINING HYPERPARAMS-----------")
    print("Data directory: {0:s}".format(hyper_json['dataDirectory']))
    print("Reviews file: {0:s}".format(hyper_json['dataFile']))
    print("Batch size of {0:d}".format(hyper_json['batchSize']))
    print("Train ratio of {0:0.2f}".format(hyper_json['trainRatio']))
    print("Train for {0:d} epochs".format(hyper_json['numEpochs']))
    print("")
    print("Testing accuracy: ", test_acc)
    print("Mean absolute distance: ", mad)
    print("")
    print("-------------CONFUSION MATRIX-------------")
    print("")
    print(conf_matrix)

if __name__ == '__main__':
    main()
