import torch
import json
from torch.utils.data import DataLoader
from model_seq2seq import MODELS, encoderRNN, decoderRNN, attention
from bleu_eval import BLEU
import pickle
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import sys

test_feat = 'MLDS_hw2_1_data/testing_data/feat'
test_label = json.load(open('MLDS_hw2_1_data/testing_label.json'))

class test_data(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])
            
    def __len__(self):
        return len(self.avi)
    def __getitem__(self, idx):
        return self.avi[idx]

def test(test_loader, model, index2word):
    model.eval()
    final_result_array = []
    for index, batch in enumerate(test_loader):
        id, avi_feats = batch
        id, avi_feats = id, Variable(avi_feats).float()
        seq_logProb, seq_predictions = model(avi_feats, mode='inference')
        test_predictions = seq_predictions
        
        result = [[index2word[x.item()] if index2word[x.item()] != '<UNK>' else 'something' for x in s] for s in test_predictions]
        result = [' '.join(s).split('<EOS>')[0] for s in result]
        final_result = zip(id, result)
        for r in final_result:
            final_result_array.append(r)
    return final_result_array

model = torch.load('model.h5', map_location=lambda storage, loc: storage)

# dataset = test_data(test_feat)
dataset = test_data('{}'.format(sys.argv[1]))
testing_loader = DataLoader(dataset, batch_size=32, shuffle=True)

with open('index2word.pickle', 'rb') as handle:
    index2word = pickle.load(handle)

test_result = test(testing_loader, model, index2word)

output = sys.argv[2]

with open("test_output.txt", 'w') as f:
    for id, caption in test_result:
        f.write('{},{}\n'.format(id, caption))

# output = "test_output.txt"
result = {}
with open(output,'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption

bleu=[]
for label in test_label:
    video_score = []
    captions = [x.rstrip('.') for x in label['caption']]
    video_score.append(BLEU(result[label['id']],captions,True))
    bleu.append(video_score[0])
average = sum(bleu) / len(bleu)
print("BLEU: " + str(average))