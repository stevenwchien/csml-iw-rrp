import numpy as np
from dataset import Dataset

# write code to take the first 100,000 lines and store in one file


def main():
    # TODO: read in one chunk of data
    ds = Dataset(path="data",filename="yelp_academic_dataset_review")
    data_chunk = ds.read(10000)

    # TODO: either before or after the train/test split, should do some feature
    # extraction

    # TODO: split into train, test data
    train_data, test_data, val_data = train_test_split(data_chunk)

    # TODO: create a model using a package, like tensorflow, and train
    model = NLP()
    model.train(train_data, val_data)
    model.save()

    model.evaluate(test_data)

    # TODO: we might need to repeat this process many times, maybe create a
    # train script

if __name__ == '__main__':
    main()
