import torch
import torch.nn as nn
import torch.optim as optim
import json
import re
from collections import defaultdict
import os
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.autograd import Variable
import random
from scipy.special import expit
import matplotlib.pyplot as plt

# read the data
train_feat = "MLDS_hw2_1_data/training_data/feat"
train_label_path = "MLDS_hw2_1_data/training_label.json"
train_label_json = json.load(open(train_label_path, 'r'))
train_label={i['id']:i['caption'] for i in train_label_json}

pattern = r'[.,!;?]+'

def data_preprocess():
    wc = defaultdict(int)    

    for captions in train_label.values():
        for caption in captions:
            words = re.sub(pattern, ' ', caption.lower()).split()
            for word in words:
                wc[word] += 1

    word_dict = {word: count for word, count in wc.items() if count >= 3}

    tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]

    index2word = {i + len(tokens): word for i, word in enumerate(word_dict)}
    word2index = {word: i + len(tokens) for i, word in enumerate(word_dict)}

    for token, index in tokens:
        index2word[index] = token
        word2index[token] = index

    return index2word, word2index, word_dict

def sentence_process(sentence, word_dict, word2index):
    sentence = re.sub(pattern, ' ', sentence).split()
    for i in range(len(sentence)):
        if sentence[i] not in word_dict:
            sentence[i] = 3
        else:
            sentence[i] = word2index[sentence[i]]
    sentence.insert(0, 1)
    sentence.append(2)
    return sentence

def annotate(word_dict, word2index):
    annotated_caption = []
    for video in train_label_json:
        for caption in video['caption']:
            caption = sentence_process(caption, word_dict, word2index)
            annotated_caption.append((video['id'], caption))
    return annotated_caption

def video():
    avi_data = {}
    files = os.listdir(train_feat)
    i = 0
    for file in files:
        i+=1
        value = np.load(os.path.join(train_feat, file))
        avi_data[file.split('.npy')[0]] = value
    return avi_data

def minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths

class training_data(Dataset):
    def __init__(self, label_file, files_dir, word_dict, word2index):
        self.label_file = label_file
        self.files_dir = files_dir
        self.word_dict = word_dict
        self.avi = video()
        self.word2index = word2index
        self.data_pair = annotate(word_dict, word2index)
        
    def __len__(self):
        return len(self.data_pair)
    
    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000)/10000.
        return torch.Tensor(data), torch.Tensor(sentence)

class attention(nn.Module):
    def __init__(self, hidden_size):
        super(attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)

        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context

class encoderRNN(nn.Module):
    def __init__(self):
        super(encoderRNN, self).__init__()
        
        self.compress = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.4)
        self.lstm = nn.LSTM(512, 512, batch_first=True)

    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()    
        input = input.view(-1, feat_n)
        input = self.compress(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, 512)

        output, t = self.lstm(input)
        hidden_state, context = t[0], t[1]
        return output, hidden_state

class decoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim):
        super(decoderRNN, self).__init__()

        self.hidden_size = 512
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.embedding = nn.Embedding(output_size, 1024)
        self.dropout = nn.Dropout(0.4)
        self.lstm = nn.LSTM(hidden_size+word_dim, hidden_size, batch_first=True)
        self.attention = attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()
        
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_cxt = torch.zeros(decoder_current_hidden_state.size())
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        seq_logProb = []
        seq_predictions = []

        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()

        for i in range(seq_len-1):
            threshold = self.teacher_forcing_ratio(training_steps=tr_steps)
            if random.uniform(0.05, 0.995) > threshold: # returns a random float value between 0.05 and 0.995
                current_input_word = targets[:, i]  
            else: 
                current_input_word = self.embedding(decoder_current_input_word).squeeze(1)

            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, t = self.lstm(lstm_input, (decoder_current_hidden_state,decoder_cxt))
            decoder_current_hidden_state=t[0]
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions
        
    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_c= torch.zeros(decoder_current_hidden_state.size())
        seq_logProb = []
        seq_predictions = []
        assumption_seq_len = 28
        
        for i in range(assumption_seq_len-1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output,  t = self.lstm(lstm_input, (decoder_current_hidden_state,decoder_c))
            decoder_current_hidden_state=t[0]
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def teacher_forcing_ratio(self, training_steps):
        return (expit(training_steps/20 +0.85)) 

class MODELS(nn.Module):
    def __init__(self, encoder, decoder):
        super(MODELS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, avi_feat, mode, target_sentences=None, tr_steps=None):
        encoder_outputs, encoder_last_hidden_state = self.encoder(avi_feat)
        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder(encoder_last_hidden_state = encoder_last_hidden_state, encoder_output = encoder_outputs,
                targets = target_sentences, mode = mode, tr_steps=tr_steps)
        elif mode == 'inference':
            seq_logProb, seq_predictions = self.decoder.infer(encoder_last_hidden_state=encoder_last_hidden_state, encoder_output=encoder_outputs)
        return seq_logProb, seq_predictions

def calculate_loss(loss_fn, x, y, lengths):
    batch_size = len(x)
    predict_cat = None
    groundT_cat = None
    flag = True

    for batch in range(batch_size):
        predict = x[batch]
        ground_truth = y[batch]
        seq_len = lengths[batch] -1

        predict = predict[:seq_len]
        ground_truth = ground_truth[:seq_len]
        if flag:
            predict_cat = predict
            groundT_cat = ground_truth
            flag = False
        else:
            predict_cat = torch.cat((predict_cat, predict), dim=0)
            groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)

    loss = loss_fn(predict_cat, groundT_cat)
    avg_loss = loss/batch_size

    return loss

def train(model, epoch, loss_fn, parameters, optimizer, train_loader):
    model.train()
    print("Epoch: ", epoch)
    batch_losses = []
    
    for batch_idx, batch in enumerate(train_loader):
        avi_feats, ground_truths, lengths = batch
        avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)
        
        optimizer.zero_grad()
        seq_logProb, seq_predictions = model(avi_feats, target_sentences = ground_truths, mode = 'train', tr_steps = epoch)
        ground_truths = ground_truths[:, 1:]  
        loss = calculate_loss(loss_fn, seq_logProb, ground_truths, lengths)
        batch_losses.append(loss.item())
        print('Batch: ', batch_idx, ' Loss:', loss.item())
        loss.backward()
        optimizer.step()

    loss = loss.item()
    return loss, batch_losses

def main():
    index2word, word2index, word_dict = data_preprocess()
    with open('index2word.pickle', 'wb') as handle:
        pickle.dump(index2word, handle, protocol = pickle.HIGHEST_PROTOCOL)
    train_dataset = training_data(train_feat, train_label_path, word_dict, word2index)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size=64, num_workers=6, shuffle=True, collate_fn=minibatch) 

    encoder = encoderRNN()
    decoder = decoderRNN(512, len(index2word) +4, len(index2word) +4, 1024)
    model = MODELS(encoder=encoder, decoder=decoder)
    
    loss_fn = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=1e-3)
    epoch_loss = []
    total_batch_losses = []
    for epoch in range(20):
        loss, batch_losses = train(model, epoch+1, loss_fn, parameters, optimizer, train_dataloader) 
        epoch_loss.append(loss)
        total_batch_losses.extend(batch_losses)
    
    with open('epoch_loss.txt', 'w') as f:
        for item in epoch_loss:
            f.write("%s\n" % item)
    with open('bacth_loss.txt', 'w') as f:
        for item in total_batch_losses:
            f.write("%s\n" % item)
    torch.save(model, "model.h5")
    print("Training Completed")
    # Plotting the batch losses
    plt.plot(total_batch_losses, label='Batch Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Batch Loss')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()