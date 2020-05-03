from transformers import BertForSequenceClassification, BertTokenizer
import torch
import numpy as np
import pandas as pd
from torch.utils.data import SequentialSampler, DataLoader

from train_model import evaluate
from data_util import generate_dataframe, extract_features

import argparse
import time

def main():
    # parse input arguments
    parser = argparse.ArgumentParser(description='argument parsing for testing')

    parser.add_argument('--data_dir',
    default='data',
    type=str,
    help='path to data directory - default: \'data\'')

    parser.add_argument('--review',
    default='yelp_reviews_test.json',
    type=str,
    help='file name containig reviews')

    parser.add_argument('--batch_size',
    default=32,
    type=int,
    help='batch size - default: 32')

    parser.add_argument('--test_size',
    default=1000,
    type=int,
    help='test set size - default: 1000')

    clargs = parser.parse_args()
    print("")
    print("==========================================")
    print("-------------Confirm Arguments------------")
    print("==========================================")

    print("Batch size of {0:d}".format(clargs.batch_size))
    print("Dataset size of {0:d}".format(clargs.test_size))
    print("Data directory for test data: {0:s}".format(clargs.data_dir))
    print("Test reviews file: {0:s}".format(clargs.review))

    # TODO: should generate some files that are equal size, so we can standardize the results
    # generate some test data
    print("")
    print("==========================================")
    print("---------------Generate Data--------------")
    print("==========================================")
    TEST_SIZE = clargs.test_size
    BATCH_SIZE = clargs.batch_size
    path = clargs.data_dir
    fn = clargs.review # remember you must include json
    filename = path + "/" + fn
    json_reader = pd.read_json(filename, lines=True, chunksize=clargs.batch_size)

    print("Generating dataset of size {0:d}".format(TEST_SIZE))
    t0 = time.perf_counter()
    test_df = generate_dataframe(json_reader, nrows=TEST_SIZE)
    elapsed = time.perf_counter() - t0
    print("Generated a dataset of size {0:d} | Took {1:0.2f} seconds".format(len(test_df), elapsed))

    # load the model from save
    print("")
    print("==========================================")
    print("----------------Load Model----------------")
    print("==========================================")

    print("Loading model and tokenizer from directory")
    model_path = './model_save/'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print("Tokenizing the data to be tested")
    dataset = extract_features(test_df, tokenizer)
    test_dataloader = DataLoader(
    dataset,
    sampler = SequentialSampler(dataset),
    batch_size = BATCH_SIZE,
    drop_last = False)

    # test the model against some test data
    print("")
    print("==========================================")
    print("----------------Test Model----------------")
    print("==========================================")
    print("Testing - Split {0:d} examples into {1:d} batches".format(TEST_SIZE, len(test_dataloader)))
    test_loss, test_acc, pred_labels, actual_labels = evaluate(model, device, test_dataloader, TEST_SIZE)
    print("")
    print("==========================================")
    print("---------------TEST RESULTS---------------")
    print("==========================================")
    print("")
    print("Testing accuracy: ", test_acc)
    print("")
    print("-------------CONFUSION MATRIX-------------")
    print("")
    # use another score, like F1 or confusion

    pred_labels = pred_labels.astype(int) + 1
    actual_labels = actual_labels.astype(int) + 1
    labels = np.stack((pred_labels, actual_labels), axis=-1)
    labels_df = pd.DataFrame(data=labels, columns=['predicted', 'actual'])
    print(pd.crosstab(labels_df['predicted'], labels_df['actual'], rownames=['Predicted'], colnames=['Actual']))

if __name__ == '__main__':
    main()
