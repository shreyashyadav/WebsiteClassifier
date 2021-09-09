import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading the dataset for Normal as well as Porn sites
df_porn = pd.read_csv("D:/python/WebsiteURLsourcecode/Classifier/Code/New folder/PornSites.csv", index_col=None)
df_normal = df = pd.read_csv("D:/python/WebsiteURLsourcecode/Classifier/Code/New folder/NormalSites.csv", index_col=None)

# Removing missing values from both datasets
df_porn.dropna(how='any', inplace=True)
df_normal.dropna(how='any', inplace=True)

# Combining both datasets
dataset = pd.concat([df_porn, df_normal], ignore_index=True)


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 2500
# This is fixed.
EMBEDDING_DIM = 100

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding,SpatialDropout1D,LSTM, Dense
from keras.callbacks import  EarlyStopping

from keras import Sequential

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(dataset['Text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


X = tokenizer.texts_to_sequences(dataset['Text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
# print('Shape of data tensor:', X.shape)

y = dataset.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.10, random_state = 42)
# print(X_train.shape,Y_train.shape)
# print(X_test.shape,Y_test.shape)


model = Sequential()
# model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
# model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 1
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
