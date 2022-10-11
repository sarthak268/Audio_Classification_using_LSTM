# Audio Classification using LSTM

Classification of Urban Sound Audio Dataset using LSTM-based model.

### Requirements
```
- pytorch==1.0.1
- scipy==1.2.0
- torchvision==0.2.1
- pandas==0.24.1
- numpy==1.14.3
- torchaudio==0.2
- librosa==0.6.3
- pydub==0.23.1
```
### Steps to follow for testing on your Test Data

- Create a folder named data/test in the current directory which will contain all the <b>'.wav'</b> files that are to be tested.

- Download 'bestModel.pt' from this <a href="https://drive.google.com/open?id=1oUWUiUr-3AIB8c1BOZcFfHgdBEVHaFLC">Link</a> and place in the current directory.

- Run the following commands:
```
python preprocess.py
python eval.py
```

- A csv file named <b>'test_predictions.csv'</b> will be generated in the current directory containing all the test files along with their corresponding predicted labels. 

### Citation

In case you find any of this useful, consider citing:

```
@misc{audio-classification-using-LSTM,
  author = {Shagun Uppal, Anish Madan, Sarthak Bhagat},
  title = {sarthak268/Audio_Classification_using_LSTM},
  url = {https://github.com/sarthak268/Audio_Classification_using_LSTM},
  year = {2019}
}
```

### Team
- Anish Madan
- Sarthak Bhagat
- Shagun Uppal  
