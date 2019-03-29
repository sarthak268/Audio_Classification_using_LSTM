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
parser.add_argument('--batch_size', type=int, default=50, help="batch size for training")
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

FLAGS = parser.parse_args()

# Hyper Parameters

# hidden_size = 60
# num_layers = 2
# num_classes = 10
# batch_size = 100
# num_epochs = 10
# learning_rate = 0.001
# data_Size = 3477


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


def extract_features(sound_clip, label, bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    for (start,end) in windows(sound_clip,window_size):
        if(len(sound_clip[int(start):int(end)]) == window_size):
            signal = sound_clip[int(start):int(end)]
            melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
            mfcc = librosa.feature.mfcc(sound_clip, label)
            logspec = librosa.core.amplitude_to_db(melspec)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)
            labels.append(label)
            
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
    return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(labels,dtype = np.int))

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


# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(RNN, self).__init__()
#         self.hidden_size = FLAGS.hidden_size
#         self.num_layers = FLAGS.num_layers
#         self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.4)#, bidirectional=True)
#         self.fc1 = nn.Linear(hidden_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(hidden_size, num_classes)
    
#     def forward(self, x):
#         x = x.float()
#         h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()).cuda() 
#         c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()).cuda()
#         out, _ = self.lstm(x, (h0, c0)) 
#         out = self.relu(self.fc1(out[:, -1, :]))
#         out = self.fc(out) 
#         return out


# =======================================================================================

# data_dir = '/home/ankitas/Desktop/DL_A2/data/train_set/fold1/'
# csv_path = '/home/ankitas/Desktop/DL_A2/data/train.csv'
# arr = os.listdir(data_dir)#+'fold1/')
# with open(os.path.join(data_dir,csv_path),'r') as f:
#     reader = csv.reader(f)
#     tr_anno_dict = dict( (k[0],k[1]) for k in reader)

# data_dir = '/home/ankitas/Desktop/DL_A2/data/test/'
# csv_path = '/home/ankitas/Desktop/DL_A2/data/test.csv'
# arr = os.listdir(data_dir)
# with open(os.path.join(data_dir,csv_path),'r') as f:
#     reader = csv.reader(f)
#     val_anno_dict = dict( (k[0],k[1]) for k in reader)

# print(len(arr))
# for i in arr:
#     print(i)
#     sound = librosa.load(os.path.join(data_dir, i))
#     label = tr_anno_dict[i.split('.')[0]]
#     features, labels = extract_features(sound, label)
#     #print(features.size())
#     dim = 7 - features.size()[0]
#     resultant_feature = torch.zeros(dim, FLAGS.bands, FLAGS.frames, FLAGS.num_features).double()
#     if(dim!=0):
#         resultant_feature = torch.cat([features, resultant_feature], 0)
#     else:
#         resultant_feature = feature
#     torch.save(resultant_feature, '/home/ankitas/Desktop/DL_A2/data/features/train/'+i.split('.')[0]+ '_'+label+'_'+(str)(dim))
#     break

# =======================================================================================

input_size = FLAGS.bands*FLAGS.frames*FLAGS.num_features
rnn = RNN(input_size, FLAGS.hidden_size, FLAGS.num_layers, FLAGS.num_classes)
#rnn = LSTM(input_size, FLAGS.hidden_size)

if(FLAGS.cuda):
    rnn.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=FLAGS.initial_learning_rate)

def loader(path):
    x = torch.load(path)
    return x

root_data_dir = './data/combined_features'

# val_data_dir = './data/features/val/'
# train_set = UrbanSoundDataset(train_data_dir)
# val_set = UrbanSoundDataset(val_data_dir)

# urban_sound_train = torch.utils.data.DataLoader(train_set, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)
# urban_sound_val = torch.utils.data.DataLoader(val_set, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)



data_sets = {x: UrbanSoundDataset(str(os.path.join(root_data_dir, x))+str('/'))
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(data_sets[x], batch_size=FLAGS.batch_size,
                                             shuffle=True, num_workers=0,drop_last=True)
              for x in ['train', 'test']}
dataset_sizes = {x: len(data_sets[x]) for x in ['train', 'test']}

writer = SummaryWriter()

# for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
#     running_loss = 0 
#     for batch_idx, (audio, label) in enumerate(urban_sound):
#         audio = audio.view(FLAGS.batch_size, -1).cuda()
#         optimizer.zero_grad()
#         predicted = rnn(audio.view(FLAGS.batch_size, FLAGS.seq_num, input_size)).cuda()
#         actual_label = torch.from_numpy(np.asarray(label)).cuda()
#         loss = criterion(predicted, actual_label)
#         running_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#         if(batch_idx%10==0):
#             print("Loss:", batch_idx, loss.item())
#     print('epoch '+ str(epoch) + ' , loss = ' + str(running_loss/(int)(FLAGS.data_Size/FLAGS.batch_size)))



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for audio, label in dataloaders[phase]:
                audio = audio.to(device)
                audio = audio.view(FLAGS.batch_size, -1)
                label = label.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(audio.view(FLAGS.batch_size, FLAGS.seq_num, input_size)).squeeze_(0)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, label)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * audio.size(0)
                running_corrects += torch.sum(preds == label.data)

                with open(FLAGS.log_file, 'a') as log:
                    log.write('{0}\t{1}\t{2}\n'.format(
                        epoch,
                        running_loss,
                        running_corrects
                    ))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if (phase == 'train'):
                writer.add_scalar('Training Loss', epoch_loss, epoch)#.data.storage().tolist()[0],epoch)
                writer.add_scalar('Training Classification Accuracy', epoch_acc, epoch)#.data.storage().tolist()[0],epoch)
            if (phase == 'test'):
                writer.add_scalar('Validation Loss', epoch_loss, epoch)#.data.storage().tolist()[0],epoch)
                writer.add_scalar('Validation Classification Accuracy', epoch_acc, epoch)#.data.storage().tolist()[0],epoch)

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model_ft = train_model(rnn.to(device), criterion, optimizer, exp_lr_scheduler,
                        num_epochs=FLAGS.end_epoch-FLAGS.start_epoch)


torch.save(model_ft,'bestModel.pt')