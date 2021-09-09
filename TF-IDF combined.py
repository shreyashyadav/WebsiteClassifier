# import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense,Dropout
from keras.callbacks import EarlyStopping
from keras import Sequential

# Loading the dataset for Normal as well as Porn sites
df_porn = pd.read_csv("D:/python/WebsiteURLsourcecode/Classifier/Code/New folder/PornSites.csv", index_col=None)
df_normal = df = pd.read_csv("D:/python/WebsiteURLsourcecode/Classifier/Code/New folder/NormalSites.csv", index_col=None)

# Removing missing values from both datasets
df_porn.dropna(how='any', inplace=True)
df_normal.dropna(how='any', inplace=True)

# Combining both datasets
dataset = pd.concat([df_porn, df_normal], ignore_index=True)

# Splitting dataset into X & y
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values

# Building a TF IDF matrix out of the corpus of reviews
td = TfidfVectorizer(max_features=10000)
X = td.fit_transform(X).toarray()
X = X[:, :, None]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

# model = Sequential()
# # model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
# # model.add(SpatialDropout1D(0.2))
# model.add(LSTM(units=100, input_shape=X_train.shape[1:], recurrent_dropout=0.2, return_sequences=True))
# model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# # model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# # model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
# # model.add(Dense(1, activation='softmax'))
# model.add(Dense(1, activation='sigmoid'))
# # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
# print(model.summary())

model = Sequential()
model.add(LSTM(units=50, input_shape=X_train.shape[1:], return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 2
batch_size = 64

# history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
# history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
# PCA
# from sklearn.decomposition import PCA
# # pca = PCA(.95)
# pca = PCA(n_components=3500)
#
# X = pca.fit_transform(X)
#
# explained_variance = pca.explained_variance_ratio_

# Splitting into training & test subsets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # Naive Bayes
# # Training the classifier & predicting on test data
# from sklearn.naive_bayes import MultinomialNB
#
# nb_classifier = MultinomialNB()
# nb_classifier.fit(X_train, y_train)
#
# y_pred = nb_classifier.predict(X_test)
#
# # Classification metrics
# classification_report = classification_report(y_test, y_pred)
#
# print('\n Accuracy for Naive bayes: ', accuracy_score(y_test, y_pred))
# print('\nClassification Report for Naive Bayes')
# print('======================================================')
# print('\n', classification_report)


# SVM
# from sklearn import svm
#
# # Creating a SVM Classifier
# svm_classifier = svm.SVC(kernel='linear')     # Linear Kernel
#
# # Training the model
# svm_classifier.fit(X_train, y_train)
#
# # Predict the response for test dataset
# y_pred1 = svm_classifier.predict(X_test)

# Classification metrics
# classification_report = classification_report(y_test, y_pred1)

# print('\n Accuracy for SVM: ', accuracy_score(y_test, y_pred1))
# #print('\nClassification Report - SVM')
# print('======================================================')
# #print('\n', classification_report)
# print('\n Confusion Matric: \n',confusion_matrix(y_test, y_pred1))
