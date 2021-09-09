import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("D:/python/WebsiteURLsourcecode/Classifier/Code/PornSites.csv",index_col=None)
df.dropna(how='any', inplace=True)

# print(df1.shape)
# print(df1.isna().sum())

sentences = df['Text']
list2 = []
word_set = []
count = 0

for word in sentences:
    list1 = str(word).split()
    list2.append(list1)

for word in list2:
    for i in range(len(word)):
        if word[i] not in word_set:
            word_set.append(word[i])
    count += 1
    print(count)

# Set of vocab
word_set = set(word_set)

# Total documents in our corpus
total_documents = len(list2)

# Creating an index for each word in our vocab.
index_dict = {}                         # Dictionary to store index for each word
i = 0
for word in word_set:
    index_dict[word] = i
    i += 1


# Create a count dictionary to keep the count of the number of documents containing the given word.
def count_dict(sentences):
    word_count = {}
    for word in word_set:
        word_count[word] = 0
        for sent in sentences:
            if word in sent:
                word_count[word] += 1
    return word_count

word_count = count_dict(list2)


# Calculating TF-IDF using TfidVectorizer
new_list = df['Text'].astype(str).values.tolist()
# print(new_list[0])

cv = TfidfVectorizer()
X = cv.fit_transform(new_list)
# Y = cv.fit_transform(new_list).toarray()

# print(cv.get_feature_names())

df1 = pd.DataFrame(X.toarray(),columns=cv.get_feature_names())
# print(df1)
# df2 = pd.DataFrame(Y,columns=cv.get_feature_names())
# print(df1)


# Calculating TF-IDF using formula
# Term Frequency
# def termfreq(document, word):
#     N = len(document)
#     occurance = len([token for token in document if token == word])
#     return occurance/N


# Inverse Document Frequency
# def inverse_doc_freq(word):
#     # try:
#     word_occurance = word_count[word]
#     # except:
#     # word_occurance = total_documents
#     return np.log(total_documents / word_occurance)


# def tf_idf(sentence):
#     tf_idf_vec = np.zeros((len(word_set),))
#     for word in sentence:
#         tf = termfreq(sentence, word)               # Calculate TF for word
#         idf = inverse_doc_freq(word)                # Calculate IDF for word
#
#         value = tf * idf
#         tf_idf_vec[index_dict[word]] = value
#     return tf_idf_vec


# #TF-IDF Encoded text corpus
# vectors = []
# for sent in list2:
#     vec = tf_idf(sent)
#     vectors.append(vec)
# print(vectors)



