import pandas as pd
import numpy as np

import argparse

# generate a dataframe that has an equal distribution of star ratings
def gen_dataset(file_reader, length):
    MAX_REVIEW_LENGTH = 120
    remaining_length = length - (length % 5)
    RATING_LENGTH = remaining_length // 5
    df_list = []
    rating_count = np.zeros(5,dtype=int)
    while remaining_length > 0:
        df_candidate = next(file_reader)
        df_candidate = df_candidate.loc[df_candidate['text'].str.split().str.len() <= MAX_REVIEW_LENGTH, ['text', 'stars', 'review_id']]
        for rating in range(1, 6, 1):
            df_rating = df_candidate[df_candidate['stars'] == rating]
            num_ratings = len(df_rating.index)
            if rating_count[rating-1] == RATING_LENGTH:
                continue
            elif num_ratings > RATING_LENGTH - rating_count[rating-1]:
                num_ratings = RATING_LENGTH - rating_count[rating-1]
                df_rating = df_rating.iloc[:num_ratings,:]
            remaining_length -= num_ratings
            rating_count[rating-1] += num_ratings
            df_list.append(df_rating)
    return pd.concat(df_list)

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

    parser.add_argument('--chunk_size',
    default=1000,
    type=int,
    help='how many samples to read at a time?')

    # parse input arguments
    clargs = parser.parse_args()
    print("")

    # name of file with reviews
    reviews_csv_path = './data/yelp_restaurant_reviews.csv'

    # use the same filereader so we dont overlap reviews
    filereader = pd.read_csv(reviews_csv_path, chunksize=clargs.chunk_size)

    # produce the largest possible testing set
    test_outfile = './data/yelp_reviews_test' + str(clargs.test_size) + '.csv'
    print("Generating Test Set of size: {0:d}".format(clargs.test_size))
    df_test = gen_dataset(filereader, clargs.test_size)
    df_test.to_csv(test_outfile)

    # produce after the test set so that the testing samples are necessarily
    # different than the training samples
    train_outfile = './data/yelp_reviews_train' + str(clargs.train_size) + '.csv'
    print("Generating Train Set of size: {0:d}".format(clargs.train_size))
    df_train = gen_dataset(filereader, clargs.train_size)
    df_train.to_csv(train_outfile)


if __name__ == '__main__':
    main()
