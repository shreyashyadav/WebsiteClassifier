# # from bs4 import BeautifulSoup
# # import re
# # import os
import nltk
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import csv
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, words
# # from wordcloud import WordCloud
# # from string import punctuation
# #
# # #nltk.download
# # lemmatizer = WordNetLemmatizer()
# # STOPWORDS = stopwords.words('english')
# # EnglishWords = words.words()
# # # print(len(EnglishWords))
# #
# # new_stopwords = ['add', 'align', 'alt', 'amp', 'app', 'arc', 'aria', 'article',
# #                  'author', 'auto', 'background', 'banner', 'blank', 'block', 'border', 'box',
# #                  'brand', 'btn', 'button', 'card', 'carousel', 'cdn', 'center', 'class',
# #                  'click', 'code', 'col', 'color', 'column', 'com', 'component', 'container', 'coupon',
# #                  'crop', 'cs', 'cta', 'custom', 'data', 'date', 'default', 'desktop', 'detail', 'display', 'div',
# #                  'document', 'dropdown', 'element', 'ellipsiscell', 'event', 'false',
# #                  'feature', 'fff', 'field', 'file', 'fill', 'filter', 'flex', 'font', 'footer', 'form', 'format',
# #                  'full', 'function', 'fusion', 'gaevent', 'grid', 'group', 'header', 'heading', 'headline',
# #                  'height', 'hidden', 'home', 'homepage', 'hover', 'href', 'html', 'http', 'https', 'icon',
# #                  'image', 'images', 'img', 'important', 'index', 'info', 'inline', 'inner', 'input',
# #                  'isarray', 'item', 'javascript', 'jpeg', 'jpg', 'key', 'label', 'layout', 'left',
# #                  'level', 'line','link', 'list', 'logo', 'main', 'margin', 'max', 'media',
# #                  'medium', 'menu', 'meta', 'min', 'mobile', 'module', 'name', 'nav', 'net', 'new', 'news',
# #                  'ngcontent', 'none', 'noscript', 'null', 'object', 'onclick', 'open', 'option', 'org', 'padding',
# #                  'page', 'path', 'pbwfnzs', 'photo', 'pic', 'picture', 'pleft', 'png', 'position', 'post',
# #                  'pright', 'primary', 'product', 'push', 'quot', 'rel', 'rem', 'return', 'rgba',
# #                  'right', 'role', 'row', 'sans', 'screen', 'script', 'search', 'section', 'self', 'share',
# #                  'site', 'slide', 'smntxt', 'social', 'source', 'spacing', 'span', 'src', 'srcset',
# #                  'start', 'static', 'story', 'style', 'sub', 'svg', 'tab', 'tabindex', 'table', 'tag', 'target',
# #                  'text', 'theme', 'thumbnail', 'tile', 'time', 'title', 'top', 'topic', 'track', 'transform',
# #                  'true', 'txt', 'type', 'typename', 'typeof', 'uitk', 'uploads', 'url', 'use', 'user',
# #                  'utm', 'value', 'var', 'video', 'view', 'viewbox', 'webkit', 'webp', 'weight', 'widget', 'width',
# #                  'window', 'wrap', 'wrapper', 'www', 'xmlns', 'zbl','datedeleted', 'datecreated', 'dateupdated',
# #                  'textcolorerrorstep', 'bgcolorerrorstep', 'textcolorsuccessstep', 'bgcolorsuccessstep',
# #                  'textcolorwarningstep', 'bgcolorwarningstep', 'bgcolorprimarystep', 'textcolorprimarystep']
# #
# # porn_stopwords = ['attribute', 'autoplay', 'backgroundcolor', 'backgroundsize', 'backgroundtype',
# #                   'backgroundwrapperblock', 'badge', 'base', 'blockbgcolor', 'blockheaderbar', 'borderroundness',
# #                   'bordersize', 'buttonalignment', 'buttonblock', 'buttonsize', 'byi', 'byq', 'byy', 'bzi', 'bzq',
# #                   'canonical', 'cardtitlelinkcolor', 'cdec', 'chat', 'closedformat', 'common', 'contain',
# #                   'contentdef', 'contentseparatorcolor', 'continuity', 'contractstar', 'day', 'defaultstate',
# #                   'dropdownsubmenubackground', 'dropdowntextcolor', 'dtd', 'ebc', 'elementor', 'enablecontinuity',
# #                   'enablecookie', 'enabled', 'end', 'expandcollapse', 'faffiliates', 'fassets', 'fbca', 'fblackbg',
# #                   'fbzday', 'fcarouselbanners', 'fcommon', 'fcontractstars', 'featuredblock',
# #                   'featuredscenelistblock', 'fexpired', 'ffffff', 'ffull', 'fid', 'fimages', 'fimageservice',
# #                   'fineeditnow', 'flogos', 'fmas', 'fmo', 'fontsizemo', 'fontsizepc', 'fonttype', 'fontweightbold',
# #                   'fpc', 'fpress', 'fpromos', 'fscenes', 'fsite', 'fsites', 'fsubsites', 'ftags', 'ftgp', 'ftour',
# #                   'ftp', 'ftrial', 'fupdate', 'fview', 'fwww', 'fzz', 'gallery', 'general', 'generic', 'gif',
# #                   'groupids', 'hasanchor', 'hasbackgroundcolor', 'hasbackgroundgradient', 'hascontrols',
# #                   'hasdropshadow', 'haslistitemsplit', 'haspaddingoverride', 'hasslider', 'hastextcolor', 'head',
# #                   'headertag', 'hoverbackgroundcolor', 'hoverbordercolor', 'hovercolor', 'httpcode',
# #                   'iconsize', 'iconstrokewidth', 'imageblock', 'instance', 'isnofollow', 'ispagination', 'isupdated',
# #                   'join', 'keywords', 'labelposition', 'lang', 'loading', 'login', 'maddos',
# #                   'manualfilter', 'matchparentheight', 'menubg', 'menuend', 'menustart', 'metadescription',
# #                   'metatags', 'metatagsconfig', 'metatitle', 'minheight', 'nbsp', 'nodename', 'nofollow',
# #                   'noopener', 'original', 'paddingmultiplier', 'pagecolorconfig', 'pagegutterconfig',
# #                   'pageskinconfig', 'parentid', 'parentstructure', 'pattern', 'php', 'playerblocks',
# #                   'popunder', 'poster', 'preview', 'priority', 'project', 'range',
# #                   'rating', 'ratio', 'recent', 'redirects', 'releasetype', 'resultslimit', 'resultsperpage',
# #                   'review', 'route', 'routetype', 'rte', 'rtecontent', 'sibling', 'sort', 'spaced',
# #                   'secondaryfont', 'segment', 'shouldopennewtab', 'showmorebutton', 'status',
# #                   'spacingmultiplier', 'string', 'structure', 'structureconfigs', 'tagids',
# #                   'tgp', 'themeconfig', 'transitiontime', 'usecustomcolors',
# #                   'usecustomskin', 'usepagegutter', 'verticalalign', 'verticalpadding', 'xml']
# #
# #
# # STOPWORDS.extend(new_stopwords)
# # STOPWORDS.extend(porn_stopwords)
# # #print(STOPWORDS)
# #
# # os.chdir("D:/python/WebsiteURLsourcecode/Normal Websites/SourceCode files")
# # # os.chdir("D:/python/WebsiteURLsourcecode/Classifier/Code/New folder")
# #
# #
# # # Creating a DataFrame and Defining an emtpy list
# # df1 = pd.DataFrame(columns=["Text"])
# # text_lst1 = []
# # sentences = []
# # word_set = []
# # index=0
# #
# # for filename in os.listdir(os.getcwd()):
# #     with open(os.path.join(os.getcwd(), filename), 'r', encoding='utf8') as f:
# #         print(filename)
# #         text = f.read()
# #
# #         text = re.sub('^[^<]+', "", text)  # remove top headers from the file
# #         data_bs = BeautifulSoup(text,'html.parser')
# #         text = data_bs.get_text()
# #
# #         #table_ = str.maketrans('', '', punctuation)
# #         #text = text.translate(table_)
# #         text = re.sub(r'[^A-Za-z]+', ' ',text)
# #         #text = re.sub(r'\W+', ' ', text)  # remove all punctuations from the data
# #
# #         text = text.replace("\n", " ")  # replace new line with space
# #         text = re.sub("\d", " ", text)  # remove all digits from the text
# #         text = re.sub("[\s]{2,}", " ", text).lower()  # replace multiple space with single space
# #
# #         text_lst = text.split(" ")  # get list of words
# #
# #         #for word in text_lst:  # Creating list of lemmatized words
# #         #    if len(lemmatizer.lemmatize(word)) > 2 and word not in STOPWORDS and lemmatizer.lemmatize(word) not in STOPWORDS:
# #         #        text_lst1.append(lemmatizer.lemmatize(word))
# #
# #         text_lst = [lemmatizer.lemmatize(word) for word in text_lst if len(lemmatizer.lemmatize(word)) > 2 and
# #                     word not in STOPWORDS and lemmatizer.lemmatize(word) not in STOPWORDS]
# #
# #
# #
# #         # process text
# #         text = " ".join(text_lst)  # getting list of words
# #         df1.loc[index,"Text"] = text
# #         index += 1
# #
# #
# # #print(df1)
# # os.chdir("D:/python/WebsiteURLsourcecode/Classifier/Code")
# # df1.to_csv(path_or_buf="NormalSites.csv", index=False)  # Writing to csv file
# #
# # # DATA PREPROCESSING
# #
# # df = pd.read_csv(filepath_or_buffer="NormalSites.csv")
# # df.dropna(how="any", inplace=True)
# #
# #
# # def plot_data(df, x, y, title, xlabel=None, ylabel=None, angle=0):
# #     """
# #     This function helps to visualize the data distribution.
# #
# #     Arguments:
# #     1. df: The input pandas dataframe.
# #     2. x: The x-axis column name, or index.
# #     3. y: The y-axis column name, or index.
# #     4. title: The title of the plot.
# #     5. xlabel: The x-axis label, defaulted to None.
# #     6. ylabel: The y-axis label, defaulted to None.
# #     7. angle: The x-axis tick rotation.
# #     """
# #
# #     plt.figure(figsize=(12, 6))
# #     sns.barplot(data=df, x=x, y=y, ci=None)
# #     plt.title(title, fontsize=18)
# #     plt.xlabel(xlabel, fontsize=15)
# #     plt.ylabel(ylabel, fontsize=15)
# #     plt.xticks(rotation=angle, fontsize=12)
# #     plt.yticks(fontsize=12)
# #     plt.show()
# #
# #
# # def frequency_charts(df, wordcloud=False, top=70, title=None):
# #     word_tokens = []                            # empty list for word tokens
# #     for sentence in df:                         # iterate over each sentence
# #         word_tokens.extend(sentence.split())    # add work tokens
# #     text_nltk = nltk.Text(word_tokens)          # generate nltk text
# #     text_freq = nltk.FreqDist(text_nltk)        # get text frequency
# #
# #     top_words = text_freq.most_common(n=top)
# #
# #     # print(top_words)
# #     words_tuple, frequeny_tuple = zip(*top_words)  # words, and their frequency
# #     print(word_tokens)
# #
# #
# #
# #     plot_data(df=None, x=list(words_tuple), y=list(frequeny_tuple), xlabel="Words",
# #               ylabel="Count", title=title, angle=60)
# #
# #     if wordcloud:  # generate word cloud
# #         wordcloud = WordCloud().generate(" ".join(word_tokens))
# #         plt.figure(figsize=(12, 7))
# #         plt.imshow(wordcloud, interpolation='bilinear')
# #         plt.axis("off")
# #
# #
# # frequency_charts(df["Text"], title="Common words - Unigram")
# import nltk
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
#
#
# df = pd.read_csv("D:/python/WebsiteURLsourcecode/Classifier/Code/PornSites.csv",index_col=None)
# df.dropna(how='any', inplace=True)
#
# new_list = df['Text'].astype(str).values.tolist()
# print(new_list[0])
#
# cv = TfidfVectorizer()
# X = cv.fit_transform(new_list).toarray()
# print(X)
# # sentences = df1['Text']
# # list2 = []
# # word_set = []
# # count = 0
# #
# # for word in sentences:
# #     list1 = str(word).split()
# #     list2.append(list1)
# #
# # # sentences1 = nltk.sent_tokenize(list2)
# # # print(sentences1)
# # # corpus = []
# # # for i in range(len(sentences1)):
# # #     review = sentences1[i]
# # #     print(review)
# #     #review = ' '.join(review)
# #     #corpus.append(review)

# import pandas as pd
#
#
# df1 = pd.read_csv("D:/python/WebsiteURLsourcecode/Classifier/Code/PornSites.csv",index_col=None)
# df1.dropna(how='any', inplace=True)
#
# def frequency_charts(df, wordcloud=False, top=1000, title=None):
#     word_tokens = []                            # empty list for word tokens
#     for sentence in df:                         # iterate over each sentence
#         word_tokens.extend(sentence.split())    # add work tokens
#     text_nltk = nltk.Text(word_tokens)          # generate nltk text
#     text_freq = nltk.FreqDist(text_nltk)        # get text frequency
#
#     top_words = text_freq.most_common(n=top)
#     print(top_words)
#     words_tuple, frequency_tuple = zip(*top_words)  # words, and their frequency
#     # print(word_tokens)
#
# frequency_charts(df1["Text"], title="Common words - Unigram")


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

# Splitting dataset into X & y
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

# Building a TF IDF matrix out of the corpus of reviews
td = TfidfVectorizer(max_features=20000)
X = td.fit_transform(X).toarray()

# PCA
from sklearn.decomposition import PCA
pca = PCA(.99)
# pca = PCA(n_components=950)

X = pca.fit_transform(X)

explained_variance = pca.explained_variance_ratio_

# Splitting into training & test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


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
from sklearn import svm

# Creating a SVM Classifier
svm_classifier = svm.SVC(kernel='linear')     # Linear Kernel

# Training the model
svm_classifier.fit(X_train, y_train)

# Predict the response for test dataset
y_pred1 = svm_classifier.predict(X_test)

# Classification metrics
#classification_report = classification_report(y_test, y_pred1)

print('\n Accuracy for SVM: ', accuracy_score(y_test, y_pred1))
#print('\nClassification Report - SVM')
print('======================================================')
#print('\n', classification_report)
print('\n Confusion Matric: \n',confusion_matrix(y_test, y_pred1))


