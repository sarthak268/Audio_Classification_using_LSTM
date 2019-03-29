import matplotlib
matplotlib.use('TkAgg')
import os
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

folder = './samples_to_visualize/'
folder_contents = os.listdir(folder)

for i in folder_contents:
	x, sr = librosa.load(os.path.join(folder, i))
	plt.figure()
	librosa.display.waveplot(x, sr=sr)
	X = librosa.feature.melspectrogram(y=x, sr=sr)
	Xdb = librosa.power_to_db(abs(X), ref=np.max)
	plt.figure()
	librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
	plt.savefig('./plt/' + str(i[:-4]) + '.png')
	plt.colorbar(format='%+2.0f dB')
	plt.title('Mel spectrogram')
	plt.tight_layout()