import numpy as np 
import os
import torch
import torchaudio
import argparse 
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import csv
import pickle


device = torch.device('cuda')
parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type=bool, default=True, help="run the following code on a GPU")
parser.add_argument('--batch_size', type=int, default=1, help="batch size for training")
parser.add_argument('--seq_num', type=int, default=7)
parser.add_argument('--frames', type=int, default=41)
parser.add_argument('--bands', type=int, default=60)
parser.add_argument('--num_features', type=int, default=2)

FLAGS = parser.parse_args() 

saved_features_dir = 'data/saved_features/'

class UrbanSoundDataset(Dataset):
    def __init__(self, file_path):
        files = os.listdir(file_path)
        #self.labels = []
        self.file_names = []
        for i in files:
            #x = i.split('_')
            self.file_names.append(i)
            #self.labels.append((int)(x[1]))
        self.file_path = file_path
        
    def __getitem__(self, index):
        path = self.file_path + str(self.file_names[index]) #+ '.wav'
        sound = torch.load(path)
        soundData = sound
        return self.file_names[index], soundData #, self.labels[index]

    def __len__(self):
        return len(self.file_names)

dset = UrbanSoundDataset(os.path.join(saved_features_dir))
dataloader = torch.utils.data.DataLoader(dset,batch_size=FLAGS.batch_size,
                                             shuffle=True, num_workers=2,drop_last=True)

input_size = FLAGS.bands*FLAGS.frames*FLAGS.num_features

dataset_sizes = len(dset)

model = torch.load('bestModel.pt')
model = model.to(device)
model.eval()
my_preds = []

running_corrects = 0.0
for audio_name,audio in dataloader:
    audio = audio.to(device)
    audio = audio.view(FLAGS.batch_size, -1)
    # label = label.to(device)

    outputs = model(audio.view(FLAGS.batch_size, FLAGS.seq_num, input_size))
    _, preds = torch.max(outputs, 1)
    my_preds.append([audio_name[0].split('_')[0],int(preds)])

with open('label_to_num_map.pickle','rb') as f:
    label_to_num_map = pickle.load(f)

num_to_label_map = {y:x for x,y in label_to_num_map.items()}

with open('test_predictions.csv','w+') as f:
    writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['ID','Class'])
    for row in my_preds:
        my_row = [row[0],num_to_label_map[row[1]]]
        writer.writerow(my_row)

