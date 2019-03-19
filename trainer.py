import re
import os
import nltk
import pickle
import argparse
import pycrfsuite
import pandas as pd
import numpy as np

from collections import Counter
from nltk.tokenize import wordpunct_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

## Part 1: Convert .txt & .ann to training data format
def ann2df(ann_file):
    ann_df = []
    for data in ann_file.split("\n")[:-1]:
        data_list = data.split("\t")
        label_info = data_list[1].split(" ")
        tag = data_list[0]
        if tag.startswith("T"):
            label = label_info[0]
            start_pos = int(label_info[1])
            end_pos = int(label_info[2])
            text = data_list[2]

            if len(wordpunct_tokenize(text)) > 1:
                offset = start_pos
                space_pos = []
                text_len = 0
                if " " in text:
                    for p in re.finditer(" ", text):
                        space_pos.append(p.start())
                    for word in wordpunct_tokenize(text):
                        if text_len == 0:
                            flabel = "B-" + label
                        else:
                            flabel = "I-" + label
                        e_pos = offset + len(word)
                        ann_df.append([word, offset, e_pos, (offset, e_pos), flabel])
                        text_len += len(word)
                        if text_len in space_pos:
                            offset = e_pos + 1
                            text_len += 1
                        else:
                            offset = e_pos
            else:
                flabel = "B-" + label
                ann_df.append([text, start_pos, end_pos, (start_pos, end_pos), flabel])
    ann_df = pd.DataFrame(ann_df, columns=["Text", "StartPosition", "EndPosition", "Position", "Label"])
    return ann_df

def spans(txt):
    tokens = wordpunct_tokenize(txt)
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        yield token, (offset, offset+len(token))
        offset += len(token)

def txt2df(txt_file):
    txt_df = []
    for i in spans(txt_file):
        txt_df.append([i[0], i[1][0], i[1][1], i[1]])
    txt_df = pd.DataFrame(txt_df, columns=["Text", "StartPosition", "EndPosition", "Position"])
    return txt_df

def create_train_data(folder, td_folder):
	if not os.path.exists(td_folder):
		os.mkdir(td_folder)

	for f in next(os.walk(folder))[2]:
		if f.endswith("txt"):
			txt_file = open(folder + "/" + f, "r").read()
			ann_file = open(folder + "/" + f.split(".")[0] + ".ann", "r").read()
			txt_df = txt2df(txt_file)
			ann_df = ann2df(ann_file)
			final_df = pd.merge(txt_df, ann_df[["Label", "Position"]], how="left", on="Position")
			final_df.fillna("O", inplace=True)
			final_df = final_df.drop(columns=["StartPosition", "EndPosition", "Position"])
			final_df.to_csv(td_folder + "/" + f.split(".")[0] + ".csv")

## Part 2: Create training data based on pycrfsuite format
def word2features(doc, num):
    word = doc[num][0]
    postag = doc[num][1]
    
    # Common features
    features = [
        "bias",
        "word.lower=" + word.lower(),
        "word.isupper=%s" % word.isupper(),
        "word.istitle=%s" % word.istitle(),
		"word[-3:]=" + word[-3:],
        "word[-2:]=" + word[-2:],
		"word[:3]=" + word[:3],
		"word[:2]=" + word[:2],
        "postag=" + postag
    ]
    
    # features for words that are not at the beginning of a sentence
    if num > 0:
        word1 = doc[num-1][0]
        postag1 = doc[num-1][1]
        features.extend([
            "-1:word.lower=" + word1.lower(),
            "-1:word.istitle=%s" % word1.istitle(),
            "-1:word.isupper=%s" % word1.isupper(),
            "-1:postag=" + postag1
        ])
    else:
        features.append("BOS") # indicate that it is the "beginning of document"
    
    # features for words that are not at the end of a document
    if num < len(doc)-1:
        word1 = doc[num+1][0]
        postag1 = doc[num+1][1]
        features.extend([
            "+1:word.lower=" + word1.lower(),
            "+1:word.istitle=%s" % word1.istitle(),
            "+1:word.isupper=%s" % word1.isupper(),
            "+1:postag=" + postag1
        ])
    else:
        features.append("EOS") # indicate it is the "end of sentence"
    
    return features

def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

def get_labels(doc):
    return [label for (token, postag, label) in doc]

def traincsv2data(td_folder):
	data = []
	for f in next(os.walk(td_folder))[2]:
		temp = pd.read_csv(td_folder + "/" + f, index_col="Unnamed: 0")
		temp = temp.dropna()
		data.append(temp)

	data_tuple = []
	for article in data:
		temp = []
		for i in article.itertuples():
			temp.append((i.Text, i.Label))
		data_tuple.append(temp)

	# add POS-tag 
	data = []
	for i, doc in enumerate(data_tuple):
		tokens = [t for t, label in doc]
		tagged = nltk.pos_tag(tokens)
		data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])
	
	X = [extract_features(doc) for doc in data]
	y = [get_labels(doc) for doc in data]
	return X, y

## Part 3: Model Classification Report
def generate_classification_report(X_test, y_test, model_name):
	tagger = pycrfsuite.Tagger()
	tagger.open(model_name)

	y_pred = [tagger.tag(xseq) for xseq in X_test]
	labels = {
		"O": 0, 
		"B-Malware": 1, 
		"I-Malware": 2, 
		"B-Target": 3, 
		"I-Target": 4, 
		"B-Actor": 5, 
		"I-Actor": 6, 
		"B-Campaign": 7,
		"I-Campaign": 8,
	}
	predictions = np.array([labels[tag] for row in y_pred for tag in row])
	truths = np.array([labels[tag] for row in y_test for tag in row])

	print(classification_report(
		truths, predictions,
		target_names=labels.keys()
	))

# Train and save the model using all data
def create_model(X, y, parameter_file, model_name):
	trainer = pycrfsuite.Trainer(verbose=True)
	for xseq, yseq in zip(X, y):
	    trainer.append(xseq, yseq)
	if parameter_file:
		with open(parameter_file, "r") as f:
			data = f.readlines()
		parameters = dict()
		for i in data:
			p = i.split("=")
			parameters[p[0]] = float(p[1].strip())
		trainer.set_params(parameters)
	trainer.train(model_name)
	print("Done training and save as crf_final.model")

## Part 4: make it into command-line program
def argumentParser():
	parser = argparse.ArgumentParser(description="Train NER using CRF algorithms")
	parser.add_argument("-a", "--article-folder", nargs=1, help="Article Folder that stored all txt and ann file", dest="article_folder")
	parser.add_argument("-t", "--train-folder", nargs=1, help="Training Folder which will saved all the CSV file (train data) after processed", dest="train_folder")
	parser.add_argument("-c", "--create-model", nargs=1, help="Create final model using all data available", dest="create_model")
	parser.add_argument("-s", "--test-size", nargs=1, type=float, help="Create Train and Test data in pickle form", dest="test_size")
	parser.add_argument("-p", "--parameter-file", nargs=1, help="Parameter file that stored the parameter for the CRF model", dest="parameter")
	parser.add_argument("-r", "--report", nargs=1, help="Generate Classification Report", dest="report")
	parser.add_argument("-f", "--file", nargs=1, help="Train or Test File", dest="file")
	parser.add_argument("-m", "--model", nargs=1, help="Model", dest="model")
	return parser

if __name__ == "__main__":
	parser = argumentParser()
	arguments = parser.parse_args()
	
	# create train_folder which save all the csv file
	if arguments.article_folder and arguments.train_folder:
		create_train_data(arguments.article_folder[0], arguments.train_folder[0])
	# create train and test pickle file
	elif arguments.train_folder and arguments.test_size:
		X, y = traincsv2data(arguments.train_folder[0])
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=arguments.test_size[0], random_state=99)
		train_data = {"X_train": X_train, "y_train": y_train}
		test_data = {"X_test": X_test, "y_test": y_test}
		pickle.dump(train_data, open("train.p", "wb"))
		pickle.dump(test_data, open("test.p", "wb"))
		print("Created train.p and test.p")
	# create all_train pickle file
	elif arguments.train_folder:
		X, y = traincsv2data(arguments.train_folder[0])
		pickle.dump({"X_train": X, "y_train": y}, open("all_train.p", "wb"))
		print("Created all_train.p")
	# create model
	elif arguments.create_model and arguments.file and arguments.model:
		train_data = pickle.load(open(arguments.file[0], "rb"))
		X_train = train_data["X_train"]
		y_train = train_data["y_train"]
		if arguments.parameter:
			create_model(X_train, y_train, arguments.parameter[0], arguments.model[0])
		else:
			create_model(X_train, y_train, False, arguments.model[0])
	# create report
	elif arguments.report and arguments.file and arguments.model:
		test_data = pickle.load(open(arguments.file[0], "rb"))
		X_test = test_data["X_test"]
		y_test = test_data["y_test"]
		generate_classification_report(X_test, y_test, arguments.model[0])
	else:
		print("No support these operations")