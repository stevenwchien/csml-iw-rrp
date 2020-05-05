import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from logger import Logger

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from data_util import generate_dataframe, extract_features, train_val_split

import time
import argparse
import os
import sys

#======================= TRAIN METHOD =======================#
def train(model, device, train_dataloader, optimizer, scheduler):
    t0 = time.perf_counter()
    total_train_loss = 0
    for step, batch in tqdm(enumerate(train_dataloader)):

        # Progress update every n batches.
        if step % 5 == 0 and not step == 0:
            elapsed = time.perf_counter() - t0
            m,s = divmod(elapsed, 60)
            print('  Batch {0:4d}  of  {1:4d}. Epoch elapsed: {2:02d}:{3:02.2f}'.format(step, len(train_dataloader), int(m), s))

        # As we unpack the batch, move to gpu
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # clear gradients
        model.zero_grad()

        # Perform a forward pass, then backward pass for gradients
        loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step gradient and step learningn rate
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    return avg_train_loss

# helper function - calculate accuracy by comparing predicted labels to actual labels
def acc(preds, labels):
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#==================== EVALUATE METHOD ====================#
def evaluate(model, device, dataloader, test_size):
    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    pred_labels = np.empty(test_size)
    actual_labels = np.empty(test_size)

    # Evaluate data for one epoch
    count = 0
    for batch in tqdm(dataloader):

        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():

            # Forward pass, calculate logit predictions.
            loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # accumulate the prediction labels and return them
        pred_labels[count*dataloader.batch_size:(count+1)*dataloader.batch_size] = np.argmax(logits, axis=1).flatten()
        actual_labels[count*dataloader.batch_size:(count+1)*dataloader.batch_size] = label_ids.flatten()
        count += 1

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

    # Report the final accuracy for this validation run.
    total_eval_accuracy = acc(pred_labels, actual_labels)
    avg_val_accuracy = total_eval_accuracy / len(dataloader)
    avg_val_loss = total_eval_loss / len(dataloader)

    return avg_val_loss, avg_val_accuracy, pred_labels, actual_labels

#==================== START MAIN METHOD ====================#
def main():
    parser = argparse.ArgumentParser(description='argument parsing for training')

    parser.add_argument('--data_dir',
    default='data',
    type=str,
    help='path to data directory - default: \'data\'')

    parser.add_argument('--review',
    default='yelp_reviews_train.json',
    type=str,
    help='file name containig reviews')

    parser.add_argument('--batch_size',
    default=32,
    type=int,
    help='batch size - default: 32')

    parser.add_argument('--dataset_size',
    default=10000,
    type=int,
    help='train size - default: 10000')

    parser.add_argument('--train_ratio',
    default=0.85,
    type=float,
    help='train size - default: 0.85')

    parser.add_argument('--epochs',
    default=4,
    type=int,
    help='number of training epochs - default: 4')

    parser.add_argument('--model_save',
    default='./model_save/',
    type=str,
    help='directory to save model')

    # parse input arguments
    clargs = parser.parse_args()

    # log to file and stdout
    sys.stdout = Logger('train')

    print("")
    print("==========================================")
    print("-------------Confirm Arguments------------")
    print("==========================================")
    print("")

    print("Batch size of {0:d}".format(clargs.batch_size))
    print("Dataset size of {0:d}".format(clargs.dataset_size))
    print("Train ratio of {0:0.2f}".format(clargs.train_ratio))
    print("Train for {0:d} epoch".format(clargs.epochs))
    print("Data directory: {0:s}".format(clargs.data_dir))
    print("Reviews file: {0:s}".format(clargs.review))
    print("Will save model in: {0:s}".format(clargs.model_save))

    # Check to see if GPU is available
    CUDA_FLAG = False
    if torch.cuda.is_available():
        CUDA_FLAG = True
        device = torch.device("cuda")
        print('*We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        CUDA_FLAG = False
        print('*No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    print("")
    print("==========================================")
    print("---------------Process Data---------------")
    print("==========================================")
    print("")

    t0 = time.perf_counter()

    TRAIN_SIZE = int(clargs.dataset_size * clargs.train_ratio)
    VAL_SIZE = clargs.dataset_size - TRAIN_SIZE
    BATCH_SIZE = clargs.batch_size
    path = clargs.data_dir
    fn = clargs.review # remember you must include json
    filename = path + "/" + fn
    json_reader = pd.read_json(filename, lines=True, chunksize=clargs.batch_size)

    # read in data from review dataset
    print("Generating dataset of size: {0:d}".format(clargs.dataset_size))
    data_df = generate_dataframe(json_reader, nrows=clargs.dataset_size)
    elapsed = time.perf_counter() - t0
    print("Generated a dataset of size: {0:d} | Took {1:0.2f} seconds".format(len(data_df), elapsed))

    t1 = time.perf_counter()

    # create tokenizer and model from transformers
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # tokenize the data into something that BERT can use, then split
    print("Tokenizing and encoding data to be fed into BERT model")
    dataset = extract_features(data_df, tokenizer)
    elapsed = time.perf_counter() - t1
    print("Finished tokenizing | Took {0:0.2f} seconds".format(elapsed))

    t2 = time.perf_counter()

    train_dataloader, validation_dataloader = train_val_split(dataset=dataset,
                                                              batch_sz=BATCH_SIZE,
                                                              lengths=[TRAIN_SIZE, VAL_SIZE])

    elapsed = time.perf_counter() - t2
    print("Training - Split {0:d} examples into {1:d} batches".format(TRAIN_SIZE, len(train_dataloader)))
    print("Validation - Split {0:d} examples into {1:d} batches".format(VAL_SIZE, len(validation_dataloader)))
    print("Finished splitting | Took {0:0.2f} seconds".format(elapsed))

    # load a pre-trained model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels = 5,
                                                          output_attentions = False,
                                                          output_hidden_states = False)
    if CUDA_FLAG:
        model.cuda()

    optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5,
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    epochs = clargs.epochs
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    # Training statistics:
    train_losses = []
    val_losses = []
    val_accs = []

    print("")
    print("==========================================")
    print("-------------Starting training------------")
    print("==========================================")
    print("")

    # TRAINING LOOP:
    # - epoch: number of times through the entire dataset
    # - consists of a training portion: forward pass, then backward pass
    # - followed by a validation portion: evaluate model on a validation set
    start_train_time = time.perf_counter()
    for i in range(epochs):
        print("-----------------Epoch {0:d}-----------------".format(i+1))
        print("Epoch {0:d} Training Phase".format(i+1))
        # first train
        model.train() # only to put the model into train mode
        train_loss = train(model, device, train_dataloader, optimizer, scheduler)
        print("  Training Loss: {0:.2f}".format(train_loss))
        train_losses.append(train_loss)
        print("")

        # then validate
        print("Epoch {0:d} Validation Phase".format(i+1))
        val_loss, val_acc, _, _ = evaluate(model, device, validation_dataloader, VAL_SIZE)
        print("  Validation Accuracy: {0:.2f}".format(val_acc))
        print("  Validation Loss: {0:.2f}".format(val_loss))
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print("")

        elapsed_time = time.perf_counter() - start_train_time
        m, s = divmod(elapsed_time, 60)
        print("End epoch {0:d} - Time so far - {1:02d}:{2:02.2f}".format((i+1), int(m), s))
        print("")


    print("==========================================")
    print("-------------Finished training------------")
    print("==========================================")
    print("")
    total_elapsed_time = time.perf_counter() - start_train_time
    m, s = divmod(total_elapsed_time, 60)
    print("Total training time: {0:02d}:{1:02.2f}".format(int(m), s))
    print("")
    print(tabulate(np.stack((train_losses, val_losses, val_accs),axis=-1), ["train_loss", "val_loss", "val_acc"]))
    print("")

    # save model
    output_dir = clargs.model_save

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Finished saving model")

if __name__ == '__main__':
    main()
