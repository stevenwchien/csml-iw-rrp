# csml-iw-rrp
for csml independent work project

## Data

This project makes use of datasets that are derived from the Yelp Open Dataset. Information about the Yelp Open Dataset can be found at: https://www.yelp.com/dataset.

Because the dataset is extremely large, we decide to create our own smaller datasets, which we describe below.

### yelp_reviews_test1000.json

This is a test dataset with 1000 reviews. For each star rating, there are 200 reviews.

### yelp_reviews_test10000.json

This is a test dataset with 10000 reviews. For each star rating, there are 200 reviews.

## Logging

Running train_model.py and test_model.py will print to both stdout and to a log file. The file name will indicate whether the run is for training the model or testing the model. The information in the training file will include elapsed run time, hyperparameters, and validation loss and accuracy.

The testing file will include elapsed time and various measures of accuracy, including just a straight up accuracy, mean absolute distance, and confusion matrix.

## Training
