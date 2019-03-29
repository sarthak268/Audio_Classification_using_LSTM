import os
import csv
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import librosa
import torchvision
from torchvision.utils import save_image
import torch.nn.functional as F
import argparse
from torch.optim import lr_scheduler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import time
import copy
from RNNs import *
from lstm import *

from tensorboardX import SummaryWriter

device = torch.device('cuda')
parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type=bool, default=True, help="run the following code on a GPU")
parser.add_argument('--batch_size', type=int, default=1, help="batch size for training")
parser.add_argument('--num_workers', type=int, default=4, help="number of workers for training")
parser.add_argument('--initial_learning_rate', type=float, default=0.00001, help='starting learning rate')
parser.add_argument('--model_save', type=str, default='model', help="save the model")
parser.add_argument('--log_file', type=str, default='log.txt', help="text file to save training logs")
parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")
parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=100, help="flag to indicate the final epoch of training")

parser.add_argument('--frames', type=int, default=41)
parser.add_argument('--bands', type=int, default=60)
parser.add_argument('--num_features', type=int, default=2)
parser.add_argument('--seq_num', type=int, default=7)

parser.add_argument('--hidden_size', type=int, default=5300, help="hidden size for lstm")
parser.add_argument('--num_layers', type=int, default=1, help="no. of layers for lstm")
parser.add_argument('--num_classes', type=int, default=10, help="number of classes for classification")
# parser.add_argument('--data_Size', type=int, default=3477, help="size of the dataset")

class UrbanSoundDataset(Dataset):
    def __init__(self, file_path):
        files = os.listdir(file_path)
        self.labels = []
        self.file_names = []
        for i in files:
            x = i.split('_')
            self.file_names.append(i)
            self.labels.append((int)(x[1]))
        self.file_path = file_path
        
    def __getitem__(self, index):
        path = self.file_path + str(self.file_names[index]) #+ '.wav'
        sound = torch.load(path)
        soundData = sound
        return soundData, self.labels[index]

    def __len__(self):
        return len(self.file_names)


FLAGS = parser.parse_args()

root_data_dir = './data/features'

input_size = FLAGS.bands*FLAGS.frames*FLAGS.num_features
rnn = RNN(input_size, FLAGS.hidden_size, FLAGS.num_layers, FLAGS.num_classes)

if(FLAGS.cuda):
    rnn.cuda()

def loader(path):
    x = torch.load(path)
    return x

rnn = loader('./bestModel.pt')
rnn.eval()

set1 = 'train'

data_sets = {x: UrbanSoundDataset(str(os.path.join(root_data_dir, x))+str('/'))
                  for x in [set1]}
dataloaders = {x: torch.utils.data.DataLoader(data_sets[x], batch_size=FLAGS.batch_size,
                                             shuffle=True, num_workers=0,drop_last=True)
              for x in [set1]}
dataset_sizes = {x: len(data_sets[x]) for x in [set1]}
phase = set1

audios = []
classes = []
count = 0

for audio, label in dataloaders[phase]:
	count += 1

	audio = audio.to(device)
	audio = audio.view(FLAGS.batch_size, -1)
	label = label.to(device)

	x = audio.view(FLAGS.batch_size, FLAGS.seq_num, input_size)
	# x = x.float()
	# h0 = Variable(torch.zeros(rnn.num_layers, x.size(0), rnn.hidden_size).float()).cuda() 
	# c0 = Variable(torch.zeros(rnn.num_layers, x.size(0), rnn.hidden_size).float()).cuda()
	# out, _ = rnn.lstm(x, (h0,c0)) 
	# out = rnn.relu(rnn.fc1(out[:, -1, :]))
	# out = rnn.relu(rnn.fc2(out))
	# out = rnn.fc3(out)
	out = rnn(x)
	out = out.squeeze_(0)
	out = out.cpu().data.numpy()
	audios.append(out)
	classes.append(label)

	#if(len(audios) == 100):
	#	break

audios = np.asarray(audios)
audios = TSNE(n_components=2, perplexity=30).fit_transform(audios)


plt.figure(figsize=(6, 5))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple']
for i in range(audios.shape[0]):
	plt.scatter(audios[i, 0], audios[i, 1], c=colors[classes[i]], s=20, marker='.')
plt.savefig('tsne.png')
    
