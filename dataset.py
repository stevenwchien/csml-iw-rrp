import json
import sys

# TODO: class that handles loading in json - can we make it so that we can
# instantiate the object, and then call it over and over again to read where
# it left off?
class Dataset:
    def __init__(self,path="data",filename="yelp_academic_dataset_review"):
        self.filename = path + "/" + filename
        print("Reading dataset")

# TODO: separate so we only deal with restaurant data - might be easy to use bert embeddings!
