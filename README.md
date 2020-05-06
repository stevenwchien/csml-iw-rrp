# Using BERT for Review Rating Prediction on the Yelp Open Dataset
A project for the CSML Independent Work requirement

## Dependencies

The dependencies for this project are listed in requirements.txt. Development was done on a MAC OSX. However, testing was done on a Linux OS - Ubuntu LTS 16.04. This was running as a VM on a Google Cloud Compute instance with a Tesla K80 GPU attached.

Note: The code was tested with the following python/package versions:

- python 3.7.0+
- [numpy](https://numpy.org/doc/) 1.18.3
- [pandas][https://pandas.pydata.org/docs/] 1.0.3
- [torch](https://github.com/pytorch/pytorch) 1.5.0
- [transformers](https://github.com/huggingface/transformers) 2.8.0
- [tqdm](https://github.com/tqdm/tqdm) 4.46.0
- [tabulate](https://pypi.org/project/tabulate/) 0.8.7

## Data

This project makes use of datasets that are derived from the Yelp Open Dataset. Information about the Yelp Open Dataset can be found at: https://www.yelp.com/dataset. Because both the Yelp Dataset and are derived dataset are relatively large, we do not include them in this repository. Instead, we include the scripts that were used to create them. All that is needed to run these scripts are the datasets downloaded from Yelp's website with their generic names.

Because the dataset is extremely large, we decided to create our own smaller datasets from the larger one. For our project, the most useful dataset to us is the reviews dataset: **yelp_academic_dataset_review.json**. To begin with our processing, we combine this data with the file: **yelp_academic_dataset_business.json** in order to extract just the reviews about restaurants. The script that creates this new csv document with only restaurant reviews is [restaurant_review_script.py](https://github.com/stevenwchien/csml-iw-rrp/blob/master/restaurant_review_script.py). Run using:

`python3 restaurant_review_script.py`

After doing this, analyzed the data further to decide how to distill it to create our training and test sets. For more details, please refer to the file

### yelp_reviews_test[number].csv

This is a test dataset with **number** entries. We make sure that the distribution of star ratings is equal. So, if there are 1000 entries, then there will be 200 1-star reviews.

### yelp_reviews_train[number].csv

This is a test dataset with **number** entries. We make sure that the distribution of star ratings is equal. So, if there are 1000 entries, then there will be 200 1-star reviews. The content of the train datasets and the test datasets are the same in structure, but we ensure that any pair of train, test files are mutually exclusive. To create a pair  of mutually train and test files, run:

`python3 create_data_script.py --train_size=[TRAIN_SIZE] --test_size=[TEST_SIZE]`

## Model

For this project, we use BERT. More information about this can be found on the Github, or their paper.

## Training

The training script takes in the following optional arguments:
- --data_dir: 

## Testing

## Logging

Running **train_model.py** and **test_model.py** will print to both stdout and to a log file. The file name will indicate whether the run is for training the model or testing the model. The information in the training file will include elapsed run time, hyperparameters, and validation loss and accuracy.

The testing file will include elapsed time and various measures of accuracy, including just straight up accuracy, mean absolute distance, and confusion matrix.
