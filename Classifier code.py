from bs4 import BeautifulSoup
import re
import os
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, words
from wordcloud import WordCloud
from string import punctuation

#nltk.download
lemmatizer = WordNetLemmatizer()
STOPWORDS = stopwords.words('english')
EnglishWords = words.words()
# print(len(EnglishWords))

new_stopwords = ['add', 'align', 'alt', 'amp', 'app', 'arc', 'aria', 'article',
                 'author', 'auto', 'background', 'banner', 'blank', 'block', 'border', 'box',
                 'brand', 'btn', 'button', 'card', 'carousel', 'cdn', 'center', 'class',
                 'click', 'code', 'col', 'color', 'column', 'com', 'component', 'container', 'coupon',
                 'crop', 'cs', 'cta', 'custom', 'data', 'date', 'default', 'desktop', 'detail', 'display', 'div',
                 'document', 'dropdown', 'element', 'ellipsiscell', 'event', 'false',
                 'feature', 'fff', 'field', 'file', 'fill', 'filter', 'flex', 'font', 'footer', 'form', 'format',
                 'full', 'function', 'fusion', 'gaevent', 'grid', 'group', 'header', 'heading', 'headline',
                 'height', 'hidden', 'home', 'homepage', 'hover', 'href', 'html', 'http', 'https', 'icon',
                 'image', 'images', 'img', 'important', 'index', 'info', 'inline', 'inner', 'input',
                 'isarray', 'item', 'javascript', 'jpeg', 'jpg', 'key', 'label', 'layout', 'left',
                 'level', 'line','link', 'list', 'logo', 'main', 'margin', 'max', 'media',
                 'medium', 'menu', 'meta', 'min', 'mobile', 'module', 'name', 'nav', 'net', 'new', 'news',
                 'ngcontent', 'none', 'noscript', 'null', 'object', 'onclick', 'open', 'option', 'org', 'padding',
                 'page', 'path', 'pbwfnzs', 'photo', 'pic', 'picture', 'pleft', 'png', 'position', 'post',
                 'pright', 'primary', 'product', 'push', 'quot', 'rel', 'rem', 'return', 'rgba',
                 'right', 'role', 'row', 'sans', 'screen', 'script', 'search', 'section', 'self', 'share',
                 'site', 'slide', 'smntxt', 'social', 'source', 'spacing', 'span', 'src', 'srcset',
                 'start', 'static', 'story', 'style', 'sub', 'svg', 'tab', 'tabindex', 'table', 'tag', 'target',
                 'text', 'theme', 'thumbnail', 'tile', 'time', 'title', 'top', 'topic', 'track', 'transform',
                 'true', 'txt', 'type', 'typename', 'typeof', 'uitk', 'uploads', 'url', 'use', 'user',
                 'utm', 'value', 'var', 'video', 'view', 'viewbox', 'webkit', 'webp', 'weight', 'widget', 'width',
                 'window', 'wrap', 'wrapper', 'www', 'xmlns', 'zbl','datedeleted', 'datecreated', 'dateupdated',
                 'textcolorerrorstep', 'bgcolorerrorstep', 'textcolorsuccessstep', 'bgcolorsuccessstep',
                 'textcolorwarningstep', 'bgcolorwarningstep', 'bgcolorprimarystep', 'textcolorprimarystep']

porn_stopwords = ['attribute', 'autoplay', 'backgroundcolor', 'backgroundsize', 'backgroundtype',
                  'backgroundwrapperblock', 'badge', 'base', 'blockbgcolor', 'blockheaderbar', 'borderroundness',
                  'bordersize', 'buttonalignment', 'buttonblock', 'buttonsize', 'byi', 'byq', 'byy', 'bzi', 'bzq',
                  'canonical', 'cardtitlelinkcolor', 'cdec', 'chat', 'closedformat', 'common', 'contain',
                  'contentdef', 'contentseparatorcolor', 'continuity', 'contractstar', 'day', 'defaultstate',
                  'dropdownsubmenubackground', 'dropdowntextcolor', 'dtd', 'ebc', 'elementor', 'enablecontinuity',
                  'enablecookie', 'enabled', 'end', 'expandcollapse', 'faffiliates', 'fassets', 'fbca', 'fblackbg',
                  'fbzday', 'fcarouselbanners', 'fcommon', 'fcontractstars', 'featuredblock',
                  'featuredscenelistblock', 'fexpired', 'ffffff', 'ffull', 'fid', 'fimages', 'fimageservice',
                  'fineeditnow', 'flogos', 'fmas', 'fmo', 'fontsizemo', 'fontsizepc', 'fonttype', 'fontweightbold',
                  'fpc', 'fpress', 'fpromos', 'fscenes', 'fsite', 'fsites', 'fsubsites', 'ftags', 'ftgp', 'ftour',
                  'ftp', 'ftrial', 'fupdate', 'fview', 'fwww', 'fzz', 'gallery', 'general', 'generic', 'gif',
                  'groupids', 'hasanchor', 'hasbackgroundcolor', 'hasbackgroundgradient', 'hascontrols',
                  'hasdropshadow', 'haslistitemsplit', 'haspaddingoverride', 'hasslider', 'hastextcolor', 'head',
                  'headertag', 'hoverbackgroundcolor', 'hoverbordercolor', 'hovercolor', 'httpcode',
                  'iconsize', 'iconstrokewidth', 'imageblock', 'instance', 'isnofollow', 'ispagination', 'isupdated',
                  'join', 'keywords', 'labelposition', 'lang', 'loading', 'login', 'maddos', 'manualfilter',
                  'matchparentheight', 'menubg', 'menuend', 'menustart', 'metadescription', 'metatags', 'metatagsconfig',
                  'metatitle', 'minheight', 'nbsp', 'nodename', 'nofollow', 'noopener', 'original', 'paddingmultiplier',
                  'pagecolorconfig', 'pagegutterconfig', 'pageskinconfig', 'parentid', 'parentstructure', 'pattern', 'php',
                  'playerblocks', 'popunder', 'poster', 'preview', 'priority', 'project', 'range', 'rating', 'ratio',
                  'recent', 'redirects', 'releasetype', 'resultslimit', 'resultsperpage', 'review', 'route', 'routetype',
                  'rte', 'rtecontent', 'sibling', 'sort', 'spaced', 'secondaryfont', 'segment', 'shouldopennewtab',
                  'showmorebutton', 'status', 'spacingmultiplier', 'string', 'structure', 'structureconfigs', 'tagids',
                  'tgp', 'themeconfig', 'transitiontime', 'usecustomcolors', 'usecustomskin', 'usepagegutter',
                  'verticalalign', 'verticalpadding', 'xml', 'january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december', 'ago', 'aug', 'see']




STOPWORDS.extend(new_stopwords)
STOPWORDS.extend(porn_stopwords)
#print(STOPWORDS)

os.chdir("D:/python/WebsiteURLsourcecode/Porn Sites/Porn Sites SourceCode")
# os.chdir("D:/python/WebsiteURLsourcecode/Classifier/Code/New folder")


# Creating a DataFrame and Defining an emtpy list
df1 = pd.DataFrame(columns=["Text"])
text_lst1 = []
sentences = []
word_set = []
index=0

for filename in os.listdir(os.getcwd()):
    with open(os.path.join(os.getcwd(), filename), 'r', encoding='utf8') as f:
        print(filename)
        text = f.read()

        text = re.sub('^[^<]+', "", text)  # remove top headers from the file
        data_bs = BeautifulSoup(text,'html.parser')
        text = data_bs.get_text()

        #table_ = str.maketrans('', '', punctuation)
        #text = text.translate(table_)
        text = re.sub(r'[^A-Za-z]+', ' ',text)
        #text = re.sub(r'\W+', ' ', text)  # remove all punctuations from the data

        text = text.replace("\n", " ")  # replace new line with space
        text = re.sub("\d", " ", text)  # remove all digits from the text
        text = re.sub("[\s]{2,}", " ", text).lower()  # replace multiple space with single space

        text_lst = text.split(" ")  # get list of words

        #for word in text_lst:  # Creating list of lemmatized words
        #    if len(lemmatizer.lemmatize(word)) > 2 and word not in STOPWORDS and lemmatizer.lemmatize(word) not in STOPWORDS:
        #        text_lst1.append(lemmatizer.lemmatize(word))

        text_lst = [lemmatizer.lemmatize(word) for word in text_lst if len(lemmatizer.lemmatize(word)) > 2 and
                    word not in STOPWORDS and lemmatizer.lemmatize(word) not in STOPWORDS]


        # process text
        text = " ".join(text_lst)  # getting list of words
        df1.loc[index,"Text"] = text
        index += 1

#print(df1)
os.chdir("D:/python/WebsiteURLsourcecode/Classifier/Code")
# df1.to_csv(path_or_buf="NormalSites.csv", index=False)  # Writing to csv file
df1.to_csv(path_or_buf="PornSites.csv", index=False)  # Writing to csv file


