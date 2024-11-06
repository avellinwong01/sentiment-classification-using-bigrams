import os
import re
import sys
import string
import argparse
import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)


class SentDataset(Dataset):
    """
    Define a pytorch dataset class that accepts a text path, and optionally label path and
    a vocabulary (depends on your implementation). This class holds all the data and implement
    a __getitem__ method to be used by a Python generator object or other classes that need it.

    DO NOT shuffle the dataset here, and DO NOT pad the tensor here.
    """
    def __init__(self, train_path, label_path=None, vocab=None):
        """
        Read the content of vocab and text_file
        Args:
            vocab (string): Path to the vocabulary file.
            text_file (string): Path to the text file.
        """
        self.texts = [] 
        self.labels = []
        if vocab == None: # training phase
            self.vocab = {}

            with open(train_path, encoding="utf8") as f:
                vocabIndex = 1
                for line in f: 
                    wordList = line.rstrip("\n").split(" ") # One line
                    bigramIndices = []
                    for i in range(len(wordList) - 1):
                        bigram = (wordList[i], wordList[i+1])
                        if bigram not in self.vocab: # add to vocabulary
                            self.vocab[bigram] = vocabIndex
                            bigramIndices.append(vocabIndex)
                            vocabIndex += 1
                        else: # bigram already exists in vocab
                            bigramIndices.append(self.vocab[bigram])

                    self.texts.append(bigramIndices) # vector/list corresponding to bigram indices of one line
        else: # testing phase
            self.vocab = vocab
            with open(train_path, encoding="utf8") as f:
                for line in f: 
                    wordList = line.rstrip("\n").split(" ") # One line
                    bigramIndices = []
                    for i in range(len(wordList) - 1):
                        bigram = (wordList[i], wordList[i+1])
                        if bigram not in self.vocab: # out of vocab bigram
                            bigramIndices.append(0) # use 0 as OOV bigram index
                        else: # bigram already exists in vocab
                            bigramIndices.append(self.vocab[bigram])

                    self.texts.append(bigramIndices) # vector/list corresponding to bigram indices of one line

        # print("texts", self.texts)
        if label_path is not None: 
            with open(label_path, encoding="utf8") as g:
                for line in g: 
                    label = int(line.rstrip("\n"))
                    self.labels.append(label)


    def vocab_size(self):
        """
        A function to inform the vocab size. The function returns two numbers:
            num_vocab: size of the vocabulary
        """
        return len(self.vocab) # should be based on training text vocab size
    
    def __len__(self):
        """
        Return the number of instances in the data
        """
        return len(self.texts) # Number of lines? 

    def __getitem__(self, i):
        """
        Return the i-th instance in the format of:
            (text, label)
        Text and label should be encoded according to the vocab (word_id).

        DO NOT pad the tensor here, do it at the collator function.
        """
        text = self.texts[i] # vector/list of bigram indices for ONE LINE
        label = self.labels[i] if len(self.labels) > 0 else None # labels are for each line (25000 in total for training)
        
        return text, label


class Model(nn.Module):
    """
    Define your model here
    """
    def __init__(self, num_vocab):
        super().__init__()
        # define your model attributes here
        embed_dim = 25
        hidden_dim = 32
        self.embedding = nn.Embedding(num_vocab+1, embed_dim, padding_idx=0) # Bigram index starts from 1
        self.linear1 = nn.Linear(embed_dim, hidden_dim) 
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        # define the forward function here
        x = self.embedding(x)
        mask = (x != 0)
        x_masked = x * mask      
        sum_values = x_masked.sum(dim=1)    
        count_values = mask.sum(dim=1)
        x = sum_values / count_values.clamp(min=1)  # Clamp to avoid division by zero
        # x = x.mean(dim=1) 
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x


def collator(batch):
    """
    Define a function that receives a list of (text, label) pair
    and return a pair of tensors:
        texts: a tensor that combines all the text in the mini-batch, pad with 0
        labels: a tensor that combines all the labels in the mini-batch
    """
    # At the point where the collator function is called, the input data is still 
    # in the form of a vector of bigram indices (length of vector depends on the length of the line)

    # List of (text, label) pair ==> multiple lines
    
    # print(batch)
    # Separate texts and labels
    texts = [item[0] for item in batch]

    # If labels are present, extract them; otherwise, set them to None
    if batch[0][1] is not None:
        labels = [item[1] for item in batch]
    else:
        labels = None
    
    # Find the length of the longest text in the batch
    max_len = max(len(text) for text in texts)
    
    # Pad all texts to the maximum length with 0 (assumed to be the padding index)
    padded_texts = [text + [0] * (max_len - len(text)) for text in texts]
    
    # Convert to tensors
    texts_tensor = torch.tensor(padded_texts, dtype=torch.long)  # (batch_size, max_len)
    if labels is not None: 
        labels_tensor = torch.tensor(labels, dtype=torch.float)  # (batch_size)
        return texts_tensor, labels_tensor
    else: 
        return texts_tensor


def train(model, dataset, batch_size, learning_rate, num_epoch, device='cpu', model_path=None):
    """
    Complete the training procedure below by specifying the loss function
    and optimizers with the specified learning rate and specified number of epoch.
    
    Do not calculate the loss from padding.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    # assign these variables
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            texts = data[0].to(device)
            labels = data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # do forward propagation 
            outputs = model(texts) # texts are already in bigram indices and padded

            outputs = outputs.squeeze(dim=1)
            # print("outputs", outputs)
            # print("labels", labels)

            # calculate the loss
            loss = criterion(outputs, labels)

            # do backward propagation
            loss.backward()

            # do the parameter optimization
            optimizer.step()

            # calculate running loss value for non padding
            running_loss += loss.item()

            # print loss value every 100 iterations and reset running loss
            # if step % 100 == 99:
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, step + 1, running_loss / 100))
            #     running_loss = 0.0
            if step % 10 == 9:  # Change from 100 to 10
                print(f'Epoch {epoch+1}, Step {step+1}, Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

    end = datetime.datetime.now()
    
    # define the checkpoint and save it to the model path
    # tip: the checkpoint can contain more than just the model
    checkpoint = {
              'model_state_dict': model.state_dict(),
              'vocab': dataset.vocab,
              }
    torch.save(checkpoint, model_path)

    print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))


def test(model, dataset, thres=0.5, device='cpu'):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=20, collate_fn=collator, shuffle=False)
    labels = []
    with torch.no_grad():
        for data in data_loader:
            texts = data.to(device)
            # print("input to model", texts)
            outputs = model(texts)
            # print("output of model", outputs)
            predictions = torch.squeeze((outputs > thres).int()) # Convert to binary labels (0 or 1)
            labels.extend(predictions.tolist())
    return labels


def main(args):
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)
    else:
        device_str = 'cpu'
    device = torch.device(device_str)
    
    assert args.train or args.test, "Please specify --train or --test"
    if args.train:
        assert args.label_path is not None, "Please provide the labels for training using --label_path argument"
        dataset = SentDataset(args.text_path, args.label_path)
        num_vocab = dataset.vocab_size()
        model = Model(num_vocab).to(device)
        
        # specify these hyper-parameters
        batch_size = 32
        learning_rate = 0.0008 
        num_epochs = 30 

        train(model, dataset, batch_size, learning_rate, num_epochs, device, args.model_path)
    if args.test:
        assert args.model_path is not None, "Please provide the model to test using --model_path argument"
        
        # load the checkpoint
        checkpoint = torch.load(args.model_path)

        # create the test dataset object using SentDataset class
        vocab = checkpoint['vocab']
        # print("vocab", vocab)
        dataset = SentDataset(args.text_path, vocab=vocab)

        # initialize and load the model
        num_vocab = dataset.vocab_size()
        model = Model(num_vocab).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']

        # run the prediction
        preds = test(model, dataset, 0.5, device)

        # write the output
        with open(args.output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(map(str, preds)))
    print('\n==== All done ====')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', help='path to the text file')
    parser.add_argument('--label_path', default=None, help='path to the label file')
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    parser.add_argument('--model_path', required=True, help='path to the model file during testing')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)
