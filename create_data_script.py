import pandas as pd
import numpy as np

import argparse

# generates a dataset of a specified length that has an equal distribution of
# stars
def gen_dataset(file_reader, length):
    MAX_REVIEW_LENGTH = 120
    df = None
    while True:
        df_candidate = next(file_reader)
        df_candidate = df_candidate.loc[df_candidate['text'].str.split().str.len() <= MAX_REVIEW_LENGTH, ['text', 'stars', 'review_id']]
        if df is None:
            df = df_candidate
        else:
            df = df.append(df_candidate)
            for rating in range(1, 6, 1):
                df_rating = df[df['stars'] == rating]
                if len(df_rating) > length//5:
                    df_rating = df_rating.iloc[:length//5, :]
                    df = df.loc[~(df['stars'] == rating), :]
                    df = df.append(df_rating)
                    if len(df) == length:
                        return df


def main():
    parser = argparse.ArgumentParser(description='argument parsing for training')

    parser.add_argument('--train_size',
    default=5000,
    type=int,
    help='how many training samples?')

    parser.add_argument('--test_size',
    default=10000,
    type=int,
    help='how many testing samples?')

    # parse input arguments
    clargs = parser.parse_args()
    print("")
    print("Generating Test Set of size: {0:d}".format(clargs.test_size))
    print("Generating Train Set of size: {0:d}".format(clargs.train_size))
    print("")

    # name of file with reviews
    reviews_csv_path = './data/yelp_restaurant_reviews.csv'

    # use the same filereader so we dont overlap reviews
    filereader = pd.read_csv(reviews_csv_path, chunksize=200)

    # produce the largest possible testing set
    train_outfile = './data/yelp_reviews_test' + str(clargs.test_size) + '.csv'
    df_test = gen_dataset(filereader, clargs.test_size)
    df_test.to_csv(train_outfile)

    # produce after the test set so that the testing samples are necessarily
    # different than the training samples
    train_outfile = './data/yelp_reviews_train' + str(clargs.train_size) + '.csv'
    df_train = gen_dataset(filereader, clargs.train_size)
    df_train.to_csv(train_outfile)


if __name__ == '__main__':
    main()
