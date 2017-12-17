# Quora-NLP
Project for DSGA 1011 - Representation Learning

### Requirements

```python
numpy
pandas
nltk
scikit-learn
keras
tensorflow
keras-mxnet
matplotlib
tqdm
gensim
googledrivedownloader
```

### Data

The primary source for the data is the [Quora dataset](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs). The data is located in the `data` folder. In order to obtain the embedding vectors, please download [Google Drive Python Package](https://pypi.python.org/pypi/googledrivedownloader/0.3) and run the script `Get Data.ipynb`.

### Analysis

The `Data Cleaning` folder contains the data preparation and analysis notebooks to produce the images and qualitative questions shown in the report. The `Data Preparation.ipynb` notebook also splits the data into the training and test proportions.

### Scripts

The implementation of [Parikh et. al (2016)](https://arxiv.org/pdf/1606.01933.pdf) can be found in `parikh.py`. The `utils.py` script contains utility functions for comparing each model in an apples to apples way.

### Notebooks

The notebooks contains the primary implementation of each model:

- Naive Embeddings - Logistic Regression.ipynb: Contains the L2 Distance Logistic Regression and MLP models using GLoVe and Word2Vec embeddings.
- Logistic Regression and MLP.ipynb - Contains the number of similar n-gram Logistic Regression and MLP models
- LSTM.ipynb - Contains the LSTM model proposed by [Rocktaschel et al (2015)](https://arxiv.org/pdf/1509.06664.pdf)
- LSTM with Angle-Distance.ipynb - Contains the LSTM model proposed by [Rocktaschel et al (2015)](https://arxiv.org/pdf/1509.06664.pdf) with the features proposed by [Tai, Socher et al (2015)](https://arxiv.org/pdf/1503.00075.pdf)
- Decomposable Attention - Standard.ipynb - Contains the decomposable attention model proposed by [Parikh et. al (2016)](https://arxiv.org/pdf/1606.01933.pdf)
- Decomposable Attention - Trainable Embeddings.ipynb - Contains the decomposable attention model proposed by [Parikh et. al (2016)](https://arxiv.org/pdf/1606.01933.pdf) with trainable embeddings
- Parameter Counter.ipynb - Notebook for counting the number of parameters in different models
