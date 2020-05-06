import numpy as np
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler, random_split
import torch

# # Custom Dataset class to handle reviews
# class ReviewDataset(data.Dataset):
#     def __init__(self, train_ratio, file_path):
#         self.review_df = pd.read_csv(file_path, index_col='review_id')
#         self.labels =
#         self.review_ids =
#         return
#     def __len__(self):
#         return
#
#     def __getitem__(self, index):
#         return self.review_df.loc['']

# extract tokenized versions of the text
def extract_features(df, tokenizer):
    input_ids = []
    attention_masks = []

    for text in tqdm(df['text']):
        # encode_plus can both tokenize, pad to length, and return attention mask
        encoded_dict = tokenizer.encode_plus(
                        text,
                        add_special_tokens = True,
                        max_length = 120,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # we have to subtract 1 here to turn the star ratings into class labels
    labels_array = df['stars'].to_numpy() - 1
    labels = torch.tensor(labels_array)

    # return a TensorDataset
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

# split randomly into a training and validation set and return as DataLoader objects
def train_val_split(dataset, batch_sz, lengths):
    train_dataset, val_dataset = random_split(dataset, lengths)

    # create dataloader objects to decrease load on memory
    train_dataloader = DataLoader(
    train_dataset,
    sampler = RandomSampler(train_dataset),
    batch_size = batch_sz,
    drop_last = False)

    validation_dataloader = DataLoader(
    val_dataset,
    sampler = SequentialSampler(val_dataset),
    batch_size = batch_sz,
    drop_last = False)

    return train_dataloader, validation_dataloader
