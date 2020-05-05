import numpy as np
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler, random_split
import torch

# generates a dataframe with reviews under a certain length
def generate_dataframe(json_reader, nrows, max_length = 120):
    df = None
    while True:
        df_candidate = next(json_reader)
        df_candidate = df_candidate.loc[df_candidate['text'].str.split().str.len() <= max_length, ['text', 'stars']]
        if df is None:
            df = df_candidate
        else:
            df = df.append(df_candidate)
            for rating in range(1, 6, 1):
                df_rating = df[df['stars'] == rating]
                if len(df_rating) > nrows//5:
                    df_rating = df_rating.iloc[:nrows//5, :]
                    df = df.loc[~(df['stars'] == rating), :]
                    df = df.append(df_rating)
                    if len(df) == nrows:
                        return df

# check to see if we need to specify stars and ratings as the columns with which we are interested
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
