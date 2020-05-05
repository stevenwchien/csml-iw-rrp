# csml-iw-rrp
for csml independent work project

## Data

This project makes use of datasets that are derived from the Yelp Open Dataset. Information about the Yelp Open Dataset can be found at: https://www.yelp.com/dataset.

Because the dataset is extremely large, we decided to create our own smaller datasets from the larger one. We specifically focus our work using the file: yelp_academic_dataset_reviews.json.

### yelp_reviews_test1000.json

This is a test dataset with 1000 reviews. For each star rating, there are 200 reviews.

### yelp_reviews_test10000.json

This is a test dataset with 10000 reviews. For each star rating, there are 200 reviews.

## Logging

Running train_model.py and test_model.py will print to both stdout and to a log file. The file name will indicate whether the run is for training the model or testing the model. The information in the training file will include elapsed run time, hyperparameters, and validation loss and accuracy.

The testing file will include elapsed time and various measures of accuracy, including just a straight up accuracy, mean absolute distance, and confusion matrix.

## Dependencies

The dependencies for this project are listed in requirements.txt. Development was done on a MAC OSX. However, testing was done on a Linux OS - Ubuntu LTS 16.04. This was running as a VM on a Google Cloud Compute instance with a Tesla K80 GPU attached.

Note: The code was tested with the following package versions:

python 3.7
transformers
pytorch
numpy
pandas
