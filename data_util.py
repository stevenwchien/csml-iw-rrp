import numpy as np
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler, random_split
import torch

# generate data with equal distribution of ratings, of maximum length
def generate_dataframe(json_reader, nrows, max_length = 100):
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
        # encode_plus can both tokenize and create the attention masks for us
        encoded_dict = tokenizer.encode_plus(
                        text,                      # text to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 120,          # Pad & truncate all text.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )

        # Add the encoded text to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels_array = df['stars'].to_numpy() - 1 # for class
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
