import heapq
import nltk
import numpy
import xlrd
from nltk import tokenize
from unidecode import unidecode
import pandas as pd
from nltk.stem.snowball import RomanianStemmer
from nltk.corpus import stopwords
import pickle


def xml_to_csv(input_filename, output_filename):
    """
    Function that converts a file from xml to csv
    :param input_filename:
    :param output_filename:
    :return:
    """
    workbook = xlrd.open_workbook(filename=input_filename, encoding_override='utf-8')
    sheet = workbook.sheet_by_index(0)

    with open(output_filename, 'w', encoding='utf-8') as f:

        # Print corresponding values
        for i in range(sheet.nrows):
            values = []
            for j in range(sheet.ncols):
                # Cast to string and replace comma found in values with whitespace
                values.append(unidecode(str(sheet.cell_value(i, j)).replace(',', ' ')))

            print(','.join(values), file=f)


def drop_invalid_lines():
    """
    Function that removes the lines with not enough information
    and renames the columns using one-word names
    :return: the new data frame, that is filtered
    """
    df = pd.read_csv("test.csv")
    pd.options.display.width = 0
    to_drop = ['institutia sursa', 'data rezultat testare',
               'data debut simptome declarate',
               'data internare',
               'sex',
               'varsta'
               ]
    df.drop(to_drop, inplace=True, axis=1)
    df = df[df['rezultat testare'].notnull()]
    df = df[df['rezultat testare'] != 'NECONCLUDENT']
    df['simptome'] = df['simptome declarate'].str.cat(df['simptome raportate la internare'], sep=' ')
    df.drop('simptome declarate', inplace=True, axis=1)
    df.drop('simptome raportate la internare', inplace=True, axis=1)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df.applymap(lambda s: s.upper() if type(s) == str else s)

    df.rename(columns={'diagnostic si semne de internare': 'diagnostic',
                       'istoric de calatorie': 'istoric',
                       'mijloace de transport folosite': 'transport',
                       'confirmare contact cu o persoana infectata': 'contact',
                       'rezultat testare': 'rezultat'
                       }, inplace=True)
    df.to_csv('test2.csv')

    return df


def preprocess_data(df):
    """
      Function that merges all the columns into train data and
      computes the encodings used as input in the classification model
      :param df: the filtered input data frame
      :return: the encoded data frame
    """
    df = stem_and_encode(df, 'simptome', 'train', 'voc1')
    df = stem_and_encode(df, 'diagnostic', 'train2', 'voc2')
    df = stem_and_encode(df, 'istoric', 'train3', 'voc3')
    df = stem_and_encode(df, 'transport', 'train4', 'voc4')
    df = stem_and_encode(df, 'contact', 'train5', 'voc5')
    my_list = numpy.array(df.train) + numpy.array(df.train2) + numpy.array(df.train3) + numpy.array(
        df.train4) + numpy.array(df.train5)
    df['train'] = my_list
    df.drop('train2', inplace=True, axis=1)
    df.drop('train3', inplace=True, axis=1)
    df.drop('train4', inplace=True, axis=1)
    df.drop('train5', inplace=True, axis=1)
    df.rezultat = df.rezultat.replace(['NEGATIV'], 0)
    df.rezultat = df.rezultat.replace(['POZITIV'], 1)
    return df


def stem_and_encode(df, column, resulting_column, filename):
    """
    Function that computes the encoding
    :param df:
    :param column:
    :param resulting_column:
    :param filename:
    :return:
    """
    i = 0
    stemmer = RomanianStemmer()
    for word_list in df[column]:
        word_list = tokenize.word_tokenize(word_list)
        stop_words = set(stopwords.words('romanian'))
        word_list = [w for w in word_list if w not in stop_words]
        new_list = []
        for word in word_list:
            new_list.append(stemmer.stem(word))
        new_list = " ".join(new_list)
        df[column][i] = new_list
        i = i + 1

    wordfreq = {}
    for sentence in df[column]:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1

    wordfreq = {k: v for k, v in sorted(wordfreq.items(), key=lambda item: item[1], reverse=True)}
    most_freq = heapq.nlargest(300, wordfreq, key=wordfreq.get)
    with open(filename, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(most_freq, filehandle)

    sentence_vectors = []
    for sentence in df[column]:
        sentence_tokens = nltk.word_tokenize(sentence)
        sent_vec = []
        for token in most_freq:
            if token in sentence_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors.append(sent_vec)
    df[resulting_column] = sentence_vectors

    return df


def preprocess_demo(name):
    df = pd.read_csv(name)
    pd.options.display.width = 0
    to_drop = ['institutia sursa', 'data rezultat testare',
               'data debut simptome declarate',
               'data internare',
               'sex',
               'varsta'
               ]
    df.drop(to_drop, inplace=True, axis=1)
    df = df[df['rezultat testare'].notnull()]
    df = df[df['rezultat testare'] != 'NECONCLUDENT']
    df['simptome'] = df['simptome declarate'].str.cat(df['simptome raportate la internare'], sep=' ')
    df.drop('simptome declarate', inplace=True, axis=1)
    df.drop('simptome raportate la internare', inplace=True, axis=1)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df.applymap(lambda s: s.upper() if type(s) == str else s)

    df.rename(columns={'diagnostic si semne de internare': 'diagnostic',
                       'istoric de calatorie': 'istoric',
                       'mijloace de transport folosite': 'transport',
                       'confirmare contact cu o persoana infectata': 'contact',
                       'rezultat testare': 'rezultat'
                       }, inplace=True)

    df = stem_and_encode_demo(df, 'simptome', 'train', 'voc1')
    df = stem_and_encode_demo(df, 'diagnostic', 'train2', 'voc2')
    df = stem_and_encode_demo(df, 'istoric', 'train3', 'voc3')
    df = stem_and_encode_demo(df, 'transport', 'train4', 'voc4')
    df = stem_and_encode_demo(df, 'contact', 'train5', 'voc5')
    my_list = numpy.array(df.train) + numpy.array(df.train2) + numpy.array(df.train3) + numpy.array(
        df.train4) + numpy.array(df.train5)
    df['train'] = my_list
    df.drop('train2', inplace=True, axis=1)
    df.drop('train3', inplace=True, axis=1)
    df.drop('train4', inplace=True, axis=1)
    df.drop('train5', inplace=True, axis=1)
    df.rezultat = df.rezultat.replace(['NEGATIV'], 0)
    df.rezultat = df.rezultat.replace(['POZITIV'], 1)
    return df


def stem_and_encode_demo(df, column, resulting_column, vocabulary):
    with open(vocabulary, 'rb') as filehandle:
        # read the data as binary data stream
        most_freq = pickle.load(filehandle)

    sentence_vectors = []

    for sentence in df[column]:
        if pd.notna(sentence):
            sentence_tokens = nltk.word_tokenize(sentence)
        else:
            sentence_tokens = []
        sent_vec = []
        for token in most_freq:
            if token in sentence_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors.append(sent_vec)
    df[resulting_column] = sentence_vectors

    return df
