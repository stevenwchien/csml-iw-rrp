# Using BERT for Review Rating Prediction on the Yelp Open Dataset
A project for the CSML Independent Work requirement

## Dependencies

The dependencies for this project are listed in requirements.txt, and can be installed in a python virtual environment using:

`pip3 install -r requirements.txt`

Note: The code was tested with the following python/package versions:

- [python](https://www.python.org/) 3.7.0+
- [numpy](https://numpy.org/doc/) 1.18.3
- [pandas](https://pandas.pydata.org/docs/) 1.0.3
- [torch](https://github.com/pytorch/pytorch) 1.5.0
- [transformers](https://github.com/huggingface/transformers) 2.8.0
- [tqdm](https://github.com/tqdm/tqdm) 4.46.0
- [tabulate](https://pypi.org/project/tabulate/) 0.8.7

Development was done on a MAC OSX. However all model training and testing were done on Ubuntu 16.04 LTS. This was running as a VM on a Google Cloud Compute instance (n1-standard-2 (2 vCPUs, 7.5 GB memory)) with a Tesla K80 GPU attached. The above dependencies/versions do work for both operating systems, but python 3.7 had to be specifically installed onto the VM. Both the training and testing script should work without modification on either a CPU and a GPU. They are set up to automatically detect and use a GPU if one is available.

## Data

This project makes use of datasets that are derived from the Yelp Open Dataset. Information about the Yelp Open Dataset can be found at: https://www.yelp.com/dataset. Because both the Yelp Dataset and are derived dataset are relatively large, we do not include them in this repository. Instead, we include the scripts that were used to create them. All that is needed to run these scripts are the datasets downloaded from Yelp's website with their generic names. For the scripts to work with their default options, create a folder called 'data' and store the files there.

Because the dataset is extremely large, we decided to create our own smaller datasets from the larger one. For our project, the most useful dataset to us is the reviews dataset: **yelp_academic_dataset_review.json**. To begin with our processing, we combine this data with the file: **yelp_academic_dataset_business.json** in order to extract just the reviews about restaurants. The script that creates this new csv document with only restaurant reviews is [restaurant_review_script.py](https://github.com/stevenwchien/csml-iw-rrp/blob/master/restaurant_review_script.py). Run using:

`python3 restaurant_review_script.py`

After creating the csv file, we further analyzed the data to decide how to distill it to create our training and test sets. For more details, please refer to the Jupyter notebook: **data_analysis.ipynb**.

### yelp_reviews_test[number].csv

This is a test dataset with **number** entries. We make sure that the distribution of star ratings is equal. So, if there are 1000 entries, then there will be 200 1-star reviews.

### yelp_reviews_train[number].csv

This is a test dataset with **number** entries. We make sure that the distribution of star ratings is equal. So, if there are 1000 entries, then there will be 200 1-star reviews. The content of the train datasets and the test datasets are the same in structure, but we ensure that any pair of train and test files created are mutually exclusive. To create a pair of mutually exclusive train and test files, run:

`python3 create_data_script.py --train_size=[TRAIN_SIZE] --test_size=[TEST_SIZE]`

## Model

For this project, we use [BERT](https://github.com/google-research/bert) from Google Research. More information about the model can be found in their [paper](https://arxiv.org/abs/1810.04805). Essentially, BERT allows us to take advantage of pre-trained models, and simply by adding a single layer, we can fine-tune the model to our desired task. BERT has been shown to be extremely effective in a multitude of sequence processing tasks. To use BERT, we use the [transformers](https://github.com/huggingface/transformers) library, which gives us access to a host of pre-trained models, one of which is BERT. The transformers library also goes the extra step of including pre-implemented task-specific models. For our project, we use `BertForSequenceClassification`.

## Training

The training script is [train_model.py](https://github.com/stevenwchien/csml-iw-rrp/blob/master/train_model.py) and takes in the following optional arguments:
- --data_dir: The directory where data is stored (default is 'data')
- --review_file: The file to find the cleaned reviews (default uses 'yelp_reviews_train5000.csv')
- --batch_size: The size of batches to use (default is 32)
- --train_ratio: The ratio of examples to use for training (default is 0.85)
- --epochs: The number of epochs to run (default is set to 4)
- --model_save: The directory to save the model after training (default is ./model_save)
- --nolog: Option to disable logging, which is on by default

We separate our training dataset into batches using a random sampler in an attempt to remove any symmetry associated with the dataset. Once training is done, the model is automatically saved into the directory provided either as an optional argument, or the default *model_save* directory. The training script also automatically saves the hyperparameters used in the training run.

## Testing

The testing script is [test_model.py](https://github.com/stevenwchien/csml-iw-rrp/blob/master/test_model.py) and takes in the following optional arguments:
- --data_dir: The directory where data is stored (default is 'data')
- --review_file: The file to find the cleaned reviews (default uses 'yelp_reviews_test1000.csv')
- --batch_size: The size of batches to use (default is 32)
- --model_save: The directory from which to pull the saved model
- --nolog: Option to disable logging, which is on by default

The testing script is set up to automatically load a model from the same default directory that the training script saves to. The testing script will also load in a summary of the training hyperparameters. During evaluation, we sequentially sample from the test dataset because the order in which we feed the data into the model does not matter.

## Logging

Running [train_model.py](https://github.com/stevenwchien/csml-iw-rrp/blob/master/train_model.py) and [test_model.py](https://github.com/stevenwchien/csml-iw-rrp/blob/master/test_model.py) will, by default, print to both stdout and to a log file. The file name will indicate whether the run is for training the model or testing the model. The file name will also include the date and time that the run was started. The information in the training log file will include elapsed run time at several points in the running process, hyperparameters, and validation loss and accuracy.

The testing file will include elapsed time and various measures of testing accuracy, including just straight up accuracy, mean absolute error, and confusion matrix. In addition to this, the testing file will print a summary of the training hyperparameters used to train the model that is being evaluated.
