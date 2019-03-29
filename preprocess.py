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
import math
from pydub import AudioSegment

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


def extract_features(sound_clip, bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    #labels = []
    for (start,end) in windows(sound_clip,window_size):
        if(len(sound_clip[int(start):int(end)]) == window_size):
            signal = sound_clip[int(start):int(end)]
            melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
            logspec = librosa.core.amplitude_to_db(melspec)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)
            #labels.append(label)
            
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
    return torch.from_numpy(np.array(features))#, torch.from_numpy(np.array(labels,dtype = np.int))

# =============================================================================

data_dir = 'data/test'                           # directory whose features are to be computed
save_dir = 'data/equal_audio/'                   # directory path where we save audio by repeating to form same size audiolength
#csv_path = 'data/test.csv'                       # annotation file for data_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


arr = os.listdir(data_dir)
arr = arr[:10]
# with open(os.path.join(csv_path),'r') as f:
#     reader = csv.reader(f)
#     tr_anno_dict = dict( (k[0],k[1]) for k in reader)          # dict mapping with key as audio name and val as class

for i in arr:
    sound, _ = librosa.load(os.path.join(data_dir, i))
    length = librosa.get_duration(sound)
    #label = tr_anno_dict[i.split('.')[0]]
    if(length<4.0):                                         # if audio length is less than our desired 4 seconds
        repetition = math.ceil(4.0 / length)                 # number of times to repeat
        try:
            sound_clip = AudioSegment.from_wav(os.path.join(data_dir, i))
        except:
            print("BAD AUDIO INPUT!!")
        else:
            sound_new = (sound_clip*(int)(repetition))         # extend the sound clip to atleast 4 secs
            sound_new = sound_new[:4000]                       # crop sound to exactly 4 secs
            sound_new.export(os.path.join(save_dir, i), format='wav')
    else:
        try:
            sound_clip = AudioSegment.from_wav(os.path.join(data_dir, i))
        except:
            print("BAD AUDIO INPUT!!")
        else:
            sound_clip.export(os.path.join(save_dir, i), format='wav')


# =============================================================================

save_features_test = 'data/saved_features/'              # directory to save features from equal sized audio samples computed above    
if not os.path.exists(save_features_test):
    os.makedirs(save_features_test)

arr = os.listdir(save_dir)
arr = arr[:10]
# with open(os.path.join(csv_path),'r') as f:
#     reader = csv.reader(f)
#     tr_anno_dict = dict( (k[0],k[1]) for k in reader)

count = 0
for i in arr:
    count+=1
    sound, _ = librosa.load(os.path.join(save_dir, i))
    #label = tr_anno_dict[i.split('.')[0]]
    features = extract_features(sound)#, label)                   
    save_path = str(save_features_test)+str(i.split('.')[0])#+ '_'+label)
    torch.save(features, save_path)                               

# =============================================================================
