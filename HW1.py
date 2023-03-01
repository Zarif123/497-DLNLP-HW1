# Imports
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math
import time
import numpy as np
import sys
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Encoding and decoding
def decode(vocab,corpus):
    
    text = ''
    for i in range(len(corpus)):
        wID = corpus[i]
        text = text + vocab[wID] + ' '
    return(text)

def encode(words,text):
    corpus = []
    tokens = text.split(' ')
    for t in tokens:
        try:
            wID = words[t][0]
        except:
            wID = words['<unk>'][0]
        corpus.append(wID)
    return(corpus)

def read_encode(file_name,vocab,words,corpus,threshold):
    
    wID = len(vocab)
    
    if threshold > -1:
        with open(file_name,'rt', encoding='utf8') as f:
            for line in f:
                line = line.replace('\n','')
                tokens = line.split(' ')
                for t in tokens:
                    try:
                        elem = words[t]
                    except:
                        elem = [wID,0]
                        vocab.append(t)
                        wID = wID + 1
                    elem[1] = elem[1] + 1
                    words[t] = elem

        temp = words
        words = {}
        vocab = []
        wID = 0
        words['<unk>'] = [wID,100]
        vocab.append('<unk>')
        for t in temp:
            if temp[t][1] >= threshold:
                vocab.append(t)
                wID = wID + 1
                words[t] = [wID,temp[t][1]]
            
                    
    with open(file_name,'rt', encoding='utf8') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(' ')
            for t in tokens:
                try:
                    wID = words[t][0]
                except:
                    wID = words['<unk>'][0]
                corpus.append(wID)
                
    return [vocab,words,corpus]

def plot_data(x, y1, y2, xlabel, ylabel, title, color1, color2, label1, label2):
    plt.plot(x, y1, color1, label=label1)
    plt.plot(x, y2, color2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def split_bios(data, words):
  fake_val, real_val = words["[FAKE]"][0], words["[REAL]"][0]
  fake_list = (data == fake_val)
  real_list = (data == real_val)
  target_indices = (fake_list + real_list).nonzero()
  bios_list = []
  start_i = 0

  for end_i in target_indices:
    bio = data[start_i:end_i+1]
    bios_list.append(bio)

    start_i = end_i + 1

  return bios_list

def clean_bios(bios, words):
  for bio_i in range(len(bios)):
    # Remove new lines and punctuation
    new_line_idx = words[''][0]
    comma_idx = words[','][0]
    period_idx = words['.'][0]
    colon_idx = words[':'][0]
    semicolon_idx = words[';'][0]

    punctuation_indices = [new_line_idx, comma_idx, period_idx, colon_idx, semicolon_idx]

    bios[bio_i] = list(filter(lambda x: x not in punctuation_indices, bios[bio_i]))

  return bios

def create_ngrams(bios, words, ngram_size):
  ngram_list = []

  for bio_i, bio in enumerate(bios):
    bio_len = len(bio)
    i = 0
    
    while i < bio_len - ngram_size:
      start, stop = i, i+ngram_size
      context = torch.tensor(bio[start:stop]).to(device)
      label = torch.tensor(bio[stop]).to(device)

      ngram = [context, label]
      ngram_list.append(ngram)

      i += 1

  return ngram_list

def clean_data(data, words):
  new_line_idx = words[''][0]
  comma_idx = words[','][0]
  period_idx = words['.'][0]
  colon_idx = words[':'][0]
  semicolon_idx = words[';'][0]

  punctuation_indices = [new_line_idx, comma_idx, period_idx, colon_idx, semicolon_idx]

  data = list(filter(lambda x: x not in punctuation_indices, data))
  return data

def create_sequences(data, words, seq_len):
  seq_list = []
  pad_char_val = words['<unk>'][0]

  num_padding = (len(data) + 1) % seq_len
  if num_padding != 0:
    pad_array = [pad_char_val] * (len(data) - num_padding)
    data = data + pad_array

  i = 0
  while i < len(data) - seq_len:
    start, stop = i, i + seq_len
    inputs = torch.tensor(data[start:stop]).detach().to(device)
    outputs = torch.tensor(data[start+1:stop+1]).detach().to(device)

    seq = [inputs, outputs]
    seq_list.append(seq)

    i += seq_len

  return seq_list

class NgramDataset(Dataset):
    def __init__(self, ngrams_data):
        self.ngrams_data_df = pd.DataFrame(ngrams_data, columns=("context", "label"))
        self.context = self.ngrams_data_df["context"]
        self.label = self.ngrams_data_df["label"]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.context[idx], self.label[idx]
    
# FeedForward Model
class FFNN(nn.Module):
    def __init__(self, vocab, words, d_model, d_hidden, dropout, ngram_size):
        super().__init__() 
    
        # Class parameters
        self.vocab = vocab
        self.words = words
        self.vocab_size = len(self.vocab)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.ngram_size = ngram_size
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Embedding Layer
        self.input_embedding = nn.Embedding(self.vocab_size, self.d_model)

        # Linear Layers
        self.fc1 = nn.Linear(ngram_size * d_model, self.d_hidden)
        self.output_embedding = nn.Linear(self.d_hidden, self.vocab_size)

        # Nonlinear Layer
        self.activation = nn.ReLU()

        # Setting weights
        self.init_weights()
                
    # Initialize weights for foward layer
    def init_weights(self):
        weight_range = 0.1
        
        self.input_embedding.weight.data.uniform_(-weight_range, weight_range)
        self.fc1.weight.data.uniform_(-weight_range, weight_range)
        self.fc1.bias.data.zero_()

    # Forward
    def forward(self, src):
        # Embeddings are fed into the forward layer
        embeds = self.input_embedding(src).view(-1, self.d_model * self.ngram_size)
        x = self.dropout(self.activation(self.fc1(embeds)))
        x = self.output_embedding(x).view(-1, self.vocab_size)
        x = F.log_softmax(x, dim=1)
        return x
    
# LSTM Model
class LSTM(nn.Module):
    def __init__(self, vocab, words, d_model, d_hidden, n_layers, dropout_rate, seq_len):
        super().__init__()
        
        # Class Parameters
        self.vocab = vocab
        self.words = words
        self.vocab_size = len(self.vocab)
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.dropout = dropout_rate
        self.seq_len = seq_len

        # Embedding Layers
        self.input_embedding = nn.Embedding(self.vocab_size, self.d_model)

        # Linear Layer
        # self.fc1 = nn.Linear(seq_len * d_model, self.d_model)
        self.fc1 = nn.Linear(self.d_hidden, self.vocab_size)

        # LSTM Cell
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.d_hidden, num_layers=self.n_layers, dropout=self.dropout)
        
    # Forward
    def forward(self, src, hc):
        embeds = self.input_embedding(src).view(self.seq_len, -1, self.d_model)
        # embeds = self.input_embedding(src).view(-1, self.batch_size, self.d_model * self.seq_len)
        # x = self.fc1(embeds)
        # preds, hc = self.lstm(x, hc)
        preds, hc = self.lstm(embeds, hc)
        preds = preds.view(-1, self.d_hidden)
        logits = self.fc1(preds).view(-1, self.vocab_size)
        return [logits, hc]
    
    def init_weights(self):
        weight_range = 0.1
        
        self.input_embedding.weight.data.uniform_(-weight_range, weight_range)
        self.fc1.weight.data.uniform_(-weight_range, weight_range)
        self.fc1.bias.data.zero_()
    
    def detach_hidden(self, hc):
        (hidden, cell) = hc
        hidden = hidden.detach()
        cell = cell.detach()
        return [hidden, cell]
    
def ffnn_train_one_epoch(model, optimizer, criterion, scheduler, train_dataloader, valid_dataloader):

  model.train(True)
  # Training Set
  running_train_acc, running_train_loss = 0, 0
  train_num_batches = len(train_dataloader)
  
  for batch_idx, (train_contexts, train_labels) in enumerate(train_dataloader):
      model.zero_grad()

      train_logits = model(train_contexts)
      train_loss = criterion(train_logits, train_labels)
      
      running_train_loss += train_loss.item()

      # if batch_idx % 10000 == 0: 
      #   print(f"Batch: {batch_idx+1}")
      #   print(f"Loss: {running_train_loss / (batch_idx+1)}")
      #   print(f"Accuracy: {running_train_acc / (batch_idx+1)}\n")

      train_loss.backward()
      optimizer.step()

      train_preds = torch.argmax(train_logits, dim=1)
      train_acc = accuracy_score(np.array(train_labels.cpu()), np.array(train_preds.cpu()))
      running_train_acc += train_acc

  train_av_acc = running_train_acc / train_num_batches
  train_av_loss = running_train_loss / train_num_batches

  model.train(False)

  # Validation Set
  running_valid_acc, running_valid_loss = 0, 0
  valid_num_batches = len(valid_dataloader)
  for batch_idx, (valid_contexts, valid_labels) in enumerate(valid_dataloader):
    
    valid_logits = model(valid_contexts)
    valid_loss = criterion(valid_logits, valid_labels)
    running_valid_loss += valid_loss.item()

    valid_preds = torch.argmax(valid_logits, dim=1)
    valid_acc = accuracy_score(np.array(valid_labels.cpu()), np.array(valid_preds.cpu()))
    running_valid_acc += valid_acc
    

  valid_av_acc = running_valid_acc / valid_num_batches
  valid_av_loss = running_valid_loss / valid_num_batches

  scheduler.step(math.exp(valid_av_loss))

  return train_av_acc, train_av_loss, valid_av_acc, valid_av_loss

def ffnn_train_loop(model, optimizer, criterion, scheduler, train_dataloader, valid_dataloader, epochs):
  train_accuracies = []
  train_losses = []
  valid_accuracies = []
  valid_losses = []
  epochs_list = list(range(epochs))

  for i in epochs_list:
    print(f"Epoch: {i+1} /////////////////////////////////////")
    # model.train(True)
    train_av_acc, train_av_loss, valid_av_acc, valid_av_loss = ffnn_train_one_epoch(model, optimizer, criterion, scheduler, train_dataloader, valid_dataloader)
    # model.train(False)

    train_perplexity = math.exp(train_av_loss)
    valid_perplexity = math.exp(valid_av_loss)

    print(f"Train Accuracy: {train_av_acc}")
    print(f"Train Loss: {train_av_loss}")
    print(f"Train Perplexity: {train_perplexity}\n")

    print(f"Valid Accuracy: {valid_av_acc}")
    print(f"Valid Loss: {valid_av_loss}")
    print(f"Valid Perplexity: {valid_perplexity}\n")

    train_accuracies.append(train_av_acc)
    train_losses.append(train_av_loss)
    valid_accuracies.append(valid_av_acc)
    valid_losses.append(valid_av_loss)

  print(train_accuracies)
  print(train_losses)
  print(valid_accuracies)
  print(valid_losses)
  print(epochs_list)
  # Graph Accuracies and Loss
  plot_data(x=epochs_list, y1=train_accuracies, y2=valid_accuracies, xlabel="Epochs", ylabel="Accuracy", title="Accuracy", color1='r', color2='b', label1='Train', label2='Valid')
  plot_data(x=epochs_list, y1=train_losses, y2=valid_losses, xlabel="Epochs", ylabel="Loss", title="Loss", color1='r', color2='b', label1='Train', label2='Valid')

def lstm_train_one_epoch(model, optimizer, criterion, train_dataloader, valid_dataloader, batch_size, clip):

  # Training Set
  running_train_acc, running_train_loss = 0, 0
  train_num_batches = len(train_dataloader)
  
  hc = (torch.randn(model.n_layers, batch_size, model.d_hidden).to(device),
        torch.randn(model.n_layers, batch_size, model.d_hidden).to(device))

  for batch_idx, (train_inputs, train_outputs) in enumerate(train_dataloader):
    model.zero_grad()
    hc = model.detach_hidden(hc)

    train_logits, hc = model(train_inputs, hc)
    train_logits = train_logits.reshape(-1, model.vocab_size)
    train_outputs = train_outputs.reshape(-1)
    train_preds = torch.argmax(train_logits, dim=1)
    train_loss = criterion(train_logits, train_outputs)
    train_acc = accuracy_score(np.array(train_outputs.cpu()), np.array(train_preds.cpu()))

    running_train_loss += train_loss.item()
    running_train_acc += train_acc

    # if batch_idx % 1000 == 0: 
    #     print(f"Train Batch: {batch_idx+1}")
    #     print(f"Train Loss: {running_train_loss / (batch_idx+1)}")
    #     print(f"Train Accuracy: {running_train_acc / (batch_idx+1)}\n")

    train_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

  train_av_acc = running_train_acc / train_num_batches
  train_av_loss = running_train_loss / train_num_batches


  # Valid Set
  hc = (torch.randn(model.n_layers, batch_size, model.d_hidden).to(device),
        torch.randn(model.n_layers, batch_size, model.d_hidden).to(device))

  running_valid_acc, running_valid_loss = 0, 0
  valid_num_batches = len(valid_dataloader)
  for batch_idx, (valid_inputs, valid_outputs) in enumerate(valid_dataloader):
    hc = model.detach_hidden(hc)

    valid_logits, hc = model(valid_inputs, hc)
    valid_logits = valid_logits.reshape(-1, model.vocab_size)
    valid_outputs = valid_outputs.reshape(-1)
    valid_preds = torch.argmax(valid_logits, dim=1)
    valid_loss = criterion(valid_logits, valid_outputs)
    valid_acc = accuracy_score(np.array(valid_outputs.cpu()), np.array(valid_preds.cpu()))

    running_valid_loss += valid_loss.item()
    running_valid_acc += valid_acc

    # if batch_idx % 1000 == 0: 
    #     print(f"Valid Batch: {batch_idx+1}")
    #     print(f"Valid Loss: {running_valid_loss / (batch_idx+1)}")
    #     print(f"Valid Accuracy: {running_valid_acc / (batch_idx+1)}\n")

  valid_av_acc = running_valid_acc / valid_num_batches
  valid_av_loss = running_valid_loss / valid_num_batches

  return train_av_acc, train_av_loss, valid_av_acc, valid_av_loss

def lstm_train_loop(model, optimizer, criterion, train_dataloader, valid_dataloader, batch_size, clip, epochs):
  train_accuracies = []
  train_losses = []
  train_perplexities = []
  valid_accuracies = []
  valid_losses = []
  valid_perplexities = []
  epochs_list = list(range(epochs))

  for i in epochs_list:
    print(f"Epoch: {i+1} /////////////////////////////////////")
    model.train(True)
    train_av_acc, train_av_loss, valid_av_acc, valid_av_loss = lstm_train_one_epoch(model, optimizer, criterion, train_dataloader, valid_dataloader, batch_size, clip)
    model.train(False)

    train_perplexity = math.exp(train_av_loss)
    valid_perplexity = math.exp(valid_av_loss)

    print(f"Train Accuracy: {train_av_acc}")
    print(f"Train Loss: {train_av_loss}")
    print(f"Train Perplexity: {train_perplexity}\n")

    print(f"Valid Accuracy: {valid_av_acc}")
    print(f"Valid Loss: {valid_av_loss}")
    print(f"Valid Perplexity: {valid_perplexity}\n")

    train_accuracies.append(train_av_acc)
    train_losses.append(train_av_loss)
    train_perplexities.append(train_perplexity)
    valid_accuracies.append(valid_av_acc)
    valid_losses.append(valid_av_loss)
    valid_perplexities.append(valid_perplexity)

  # Graph Accuracies and Loss
  plot_data(x=epochs_list, y1=train_accuracies, y2=valid_accuracies, xlabel="Epochs", ylabel="Accuracy", title="Accuracy", color1='b', color2='r', label1='Train', label2='Valid')
  plot_data(x=epochs_list, y1=train_losses, y2=valid_losses, xlabel="Epochs", ylabel="Loss", title="Loss", color1='b', color2='r', label1='Train', label2='Valid')
  plot_data(x=epochs_list, y1=train_perplexities, y2=valid_perplexities, xlabel="Epochs", ylabel="Perplexity", title="Perplexity", color1='b', color2='r', label1='Train', label2='Valid')

def create_histogram(model, dataloader, ngram_size, words):
  bio_len = 1
  fake_bio_probs, real_bio_probs = [], []
  total_bio_prob = 0
  MAX_VAL = 25000

  fake_val, real_val = words["[FAKE]"][0], words["[REAL]"][0]  

  for batch_idx, (context, label) in enumerate(dataloader):
    probs = model(context).squeeze()
    # probs = F.softmax(probs)
    label_prob = probs[label]

    if batch_idx % 50000 == 0: 
      print(f"{batch_idx}/{len(dataloader)}")
      print(label_prob)
    if label_prob == 0:
      label_prob += 0.00000001

    # pseudo_prob = -math.log(label_prob)
    pseudo_prob = -label_prob
    total_bio_prob += pseudo_prob

    if label == fake_val:
      bio_prob = total_bio_prob / bio_len
      if bio_prob < MAX_VAL: fake_bio_probs.append(bio_prob)
      bio_len = 1
      total_bio_probs = 0
    elif label == real_val:
      bio_prob = total_bio_prob / bio_len
      if bio_prob < MAX_VAL: real_bio_probs.append(bio_prob)
      bio_len = 1
      total_bio_probs = 0
    else:
      bio_len += 1

  sns.set(style="darkgrid")
 
  print(fake_bio_probs)
  print(real_bio_probs)

  plt.figure(0)
  sns.kdeplot(np.array(fake_bio_probs))
  sns.kdeplot(np.array(real_bio_probs))
  plt.xlabel("Negative Average Log Probs")
  plt.ylabel("Density")
  plt.show()

  plt.figure(1)
  plt.hist(fake_bio_probs, label='Fakes', alpha=0.5, bins=100)
  plt.hist(real_bio_probs, label='Reals', alpha=0.5, bins=100)
  plt.legend(loc='upper left')
  plt.xlabel("Negative Average Log Probs")
  plt.ylabel("Frequency")
  plt.show()

def ffnn_test_model(model, dataloader, words, threshold):
  bio_len = 1
  truth, preds = [], []
  total_bio_prob = 0

  fake_val, real_val = words["[FAKE]"][0], words["[REAL]"][0]  

  for batch_idx, (context, label) in enumerate(dataloader):
    probs = model(context).squeeze()
    # probs = F.softmax(probs)
    label_prob = probs[label]

    if label_prob == 0:
      label_prob += 0.00000001

    # pseudo_prob = -math.log(label_prob)
    pseudo_prob = -label_prob
    total_bio_prob += pseudo_prob

    if label == fake_val or label == real_val:
      truth.append(label.item())

      bio_prob = total_bio_prob / bio_len
      if bio_prob < threshold:
        preds.append(real_val)
      else:
        preds.append(fake_val)

      bio_len = 1
      total_bio_probs = 0
    else:
      bio_len += 1

  acc = accuracy_score(np.array(truth), np.array(preds))
  confusion_mat = confusion_matrix(truth, preds)
  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=["REAL", "FAKE"])
  disp.plot()
  plt.show()
  print(f"Test Accuracy: {acc}")
    
def lstm_test_model(model, test_sequences, test_words):
  print("Starting Test and Confusion Matrix Construction")

  hc = (torch.randn(model.n_layers, 1, model.d_hidden).to(device),
        torch.randn(model.n_layers, 1, model.d_hidden).to(device))

  truth = []
  preds = []
  num_sequences = len(test_sequences)

  for idx, (input, output) in enumerate(test_sequences):
    if idx % 2000 == 0:
      print("Testing Sequence:", idx, "/", num_sequences)

    hc = model.detach_hidden(hc)

    logits, hc = model(input, hc)
    probs = torch.softmax(logits, dim=1)

    fake_val = test_words["[FAKE]"][0]
    real_val = test_words["[REAL]"][0]

    fake_list = (output == fake_val)
    real_list = (output == real_val)
    prediction_indices = (fake_list + real_list).nonzero()

    for i in prediction_indices:
      truth.append(np.array(output[i].cpu()))
      if probs[i, real_val] > probs[i, fake_val]:
        preds.append(real_val)
      else:
        preds.append(fake_val)

  acc = accuracy_score(np.array(truth), np.array(preds))
  confusion_mat = confusion_matrix(truth, preds)
  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=["REAL", "FAKE"])
  disp.plot()
  plt.show()

  return acc

class Params:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
model_map = {0: 'FFNN', 1: 'LSTM', 2: 'FFNN_CLASSIFY', 3: 'LSTM_CLASSIFY'}
train_map = {0: 'data/real.train.tok', 1: 'data/fake.train.tok', 2: 'data/mix.train.tok'}
valid_map = {0: 'data/real.valid.tok', 1: 'data/fake.valid.tok', 2: 'data/mix.valid.tok'}
test_map = {0: 'data/real.test.tok', 1: 'data/fake.test.tok', 2: 'data/mix.test.tok', 3: 'data/blind.test.tok'}

model_type = model_map[0]

# Types of data
train_type = train_map[2]
valid_type = valid_map[2]
test_type = test_map[2]

# FFNN Args:
# args = {
#     "d_model": 4,
#     "d_hidden": 4,
#     "n_layers": 3,
#     "batch_size": 20,
#     "seq_len": 30,
#     "printevery": 5000,
#     "window": 3,
#     "epochs": 30,
#     "lr": 0.00001,
#     "dropout": 0.35,
#     "clip": 2.0,
#     "model": model_type,
#     "savename": model_type.lower(),
#     "loadname": model_type.lower(),
#     "trainname": train_type,
#     "validname": valid_type,
#     "testname": test_type
# }

# LSTM Args:
args = {
    "d_model": 512,
    "d_hidden": 512,
    "n_layers": 2,
    "batch_size": 64,
    "seq_len": 30,
    "printevery": 5000,
    "window": 3,
    "epochs": 30,
    "lr": 0.00001,
    "dropout": 0.35,
    "clip": 2.0,
    "model": model_type,
    "savename": model_type.lower(),
    "loadname": model_type.lower(),
    "trainname": train_type,
    "validname": valid_type,
    "testname": test_type
}

# Main Function
def main(args): 
    torch.manual_seed(0)
    
    # params
    params = Params(**args)
    train_name = params.trainname
    valid_name = params.validname
    test_name = params.testname
    model_type = params.model
    d_model = params.d_model
    d_hidden = params.d_hidden
    dropout = params.dropout
    epochs = params.epochs
    window = params.window
    seq_len = params.seq_len
    batch_size = params.batch_size
    lr = params.lr
    n_layers = params.n_layers
    clip = params.clip

    # Extract vocab and words
    [train_vocab,train_words,train] = read_encode(train_name,[],{},[],3)
    train_data = torch.tensor(train)

    [valid_vocab,valid_words,valid] = read_encode(valid_name,[],{},[],3)
    valid_data = torch.tensor(valid)

    [test_vocab,test_words,test] = read_encode(test_name,[],{},[],3)
    test_data = torch.tensor(test)
    
    if model_type == 'FFNN':

      # Process Train Data
      train_bios = split_bios(train_data, train_words)
      train_bios = clean_bios(train_bios, train_words)
      train_ngrams_data = create_ngrams(train_bios, train_words, window)

      train_ngram_dataset = NgramDataset(train_ngrams_data)
      train_ngram_dataloader = DataLoader(train_ngram_dataset, batch_size=batch_size)

      # Process Valid Data
      valid_bios = split_bios(valid_data, valid_words)
      valid_bios = clean_bios(valid_bios, valid_words)
      valid_ngrams_data = create_ngrams(valid_bios, valid_words, window)

      valid_ngram_dataset = NgramDataset(valid_ngrams_data)
      valid_ngram_dataloader = DataLoader(valid_ngram_dataset, batch_size=batch_size)

      ngram_model = FFNN(train_vocab, train_words, d_model, d_hidden, dropout, window).to(device)
      optimizer = torch.optim.Adam(ngram_model.parameters(), lr=lr)
      criterion = nn.NLLLoss()
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=0)

      ffnn_train_loop(ngram_model, optimizer, criterion, scheduler, train_ngram_dataloader, valid_ngram_dataloader, epochs)

      torch.save(ngram_model.state_dict(), 'ffnn.pth')


    if model_type == 'LSTM':
        # Process Train Data
      clean_train_data = clean_data(train_data, train_words)
      train_sequences = create_sequences(clean_train_data, train_words, seq_len)
      train_lstm_dataset = NgramDataset(train_sequences)
      train_lstm_dataloader = DataLoader(train_lstm_dataset, batch_size=batch_size, drop_last=True)

      # Process Valid Data
      clean_valid_data = clean_data(valid_data, valid_words)
      valid_sequences = create_sequences(clean_valid_data, valid_words, seq_len)
      valid_lstm_dataset = NgramDataset(valid_sequences)
      valid_lstm_dataloader = DataLoader(valid_lstm_dataset, batch_size=batch_size, drop_last=True)

      # Create and Train LSTM Model
      lstm_model = LSTM(train_vocab, train_words, d_model, d_hidden, n_layers, dropout, seq_len).to(device)
      optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)
      criterion = nn.CrossEntropyLoss()

      lstm_train_loop(lstm_model, optimizer, criterion, train_lstm_dataloader, valid_lstm_dataloader, batch_size, clip, epochs)

      torch.save(lstm_model.state_dict(), 'lstm.pth')

    if model_type == 'FFNN_CLASSIFY':
      valid_bios = split_bios(valid_data, valid_words)
      valid_bios = clean_bios(valid_bios, valid_words)
      valid_ngrams_data = create_ngrams(valid_bios, valid_words, window)

      valid_ngram_dataset = NgramDataset(valid_ngrams_data)
      valid_ngram_dataloader = DataLoader(valid_ngram_dataset, batch_size=1)

      test_bios = split_bios(test_data, test_words)
      test_bios = clean_bios(test_bios, test_words)
      test_ngrams_data = create_ngrams(test_bios, test_words, window)

      test_ngram_dataset = NgramDataset(test_ngrams_data)
      test_ngram_dataloader = DataLoader(test_ngram_dataset, batch_size=1)

      ngram_model = FFNN(train_vocab, train_words, d_model, d_hidden, dropout, window).to(device)
      ngram_model.load_state_dict(torch.load('ffnn.pth', map_location=torch.device('cpu')))
      ngram_model.eval()
        
      create_histogram(ngram_model, valid_ngram_dataloader, window, valid_words)
      threshold = 4500
      ffnn_test_model(ngram_model, test_ngram_dataloader, test_words, threshold)
    if model_type == 'LSTM_CLASSIFY':
      loaded_lstm_model = LSTM(train_vocab, train_words, d_model, d_hidden, n_layers, dropout, seq_len).to(device)
      loaded_lstm_model.load_state_dict(torch.load('lstm4.pth', map_location=torch.device('cpu')))
      loaded_lstm_model.eval()

      # Process Test Data
      clean_test_data = clean_data(test_data, test_words)
      test_sequences = create_sequences(clean_test_data, train_words, seq_len)
      # clean_test_data = clean_data(test_data, test_words)
      # test_sequences = create_sequences(clean_test_data, test_words, seq_len)

      # Test LSTM Model
      accuracy = lstm_test_model(loaded_lstm_model, test_sequences, test_words)
      print(accuracy)


main(args)