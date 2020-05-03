from transformers import BertForSequenceClassification, BertTokenizer
import torch
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

    parser.add_argument('--dataset_size',
    default=1000,
    type=int,
    help='dataset size - default: 1000')

    clargs = parser.parse_args()

    print("==========================================")
    print("-------------Confirm Arguments------------")
    print("==========================================")

    print("Batch size of {0:3d}".format(clargs.batch_size))
    print("Dataset size of {0:3d}".format(clargs.dataset_size))
    print("Data directory for test data: {0:s}".format(clargs.data_dir))
    print("Test reviews file: {0:s}".format(clargs.review))

    # generate some test data
    print("==========================================")
    print("---------------Generate Data--------------")
    print("==========================================")
    TEST_SIZE = clargs.dataset_size
    BATCH_SIZE = clargs.batch_size
    path = clargs.data_dir
    fn = clargs.review # remember you must include json
    filename = path + "/" + fn
    json_reader = pd.read_json(filename, lines=True, chunksize=500)

    print("Generating dataset of size: {0:6d}".format(TEST_SIZE))
    t0 = time.perf_counter()
    test_df = generate_dataframe(json_reader, nrows=TEST_SIZE)
    elapsed = time.perf_counter() - t0
    print("Generated a dataset of size: {0:6d} | Took {1:0.2f} seconds".format(len(test_df), elapsed))

    # create tokenizer and model from transformers
    model_path = './model_save/'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    dataset = extract_features(test_df, tokenizer)

    test_dataloader = DataLoader(
    dataset,
    sampler = SequentialSampler(dataset),
    batch_size = BATCH_SIZE,
    drop_last = False)

    print("Testing - Split {0:d} examples into {1:d} batches".format(TEST_SIZE, len(test_dataloader)))

    # load the model from save
    print("==========================================")
    print("----------------Load Model----------------")
    print("==========================================")

    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print("loaded from save")

    # test the model against some test data
    print("==========================================")
    print("----------------Test Model----------------")
    print("==========================================")
    test_loss, test_acc = evaluate(model, device, test_dataloader)
    print("Testing accuracy: ", test_acc)

    # use another score, like F1 or confusion

if __name__ == '__main__':
    main()
