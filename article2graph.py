import nltk
import argparse
import pycrfsuite

from nltk.tokenize import wordpunct_tokenize
from trainer import extract_features
from collections import Counter
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, XSD, RDFS
from rdflib.term import Literal, URIRef

class Stack:
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return self.items == []
    def push(self, item):
        self.items.insert(0, item)
    def pop(self):
        return self.items.pop(0)
    def peek(self):
        return self.items[0]

def article2trainFormat(article):
	tokens = wordpunct_tokenize(article)
	pos_tags = nltk.pos_tag(tokens)
	return pos_tags

def model_predict(model_name, data):
	tagger = pycrfsuite.Tagger()
	tagger.open(model_name)
	return tagger.tag(data)

def gen_text_label(word_predicted):
	"""
	word_predicted has format [(word, predict)]
	"""
	text_stack = Stack()
	text_list = list()
	for word, label in word_predicted[::-1]:
		if label.startswith("B"):
			if text_stack.isEmpty():
				text_list.append((word, label[2:]))
			else:
				text_stack.push(word)
				text = ""
				while (not text_stack.isEmpty()):
					temp = text_stack.pop()
					if temp.isalpha():
						text += temp + " "
					else:
						text += temp
				text_list.append((text, label[2:]))
		elif label.startswith("I"):
			text_stack.push(word)
	return text_list

def argumentParser():
	parser = argparse.ArgumentParser(description="Train NER using CRF algorithms")
	parser.add_argument("-a", "--article-file", nargs=1, required=True, help="article file (txt) to create knowledge graph", dest="article")
	parser.add_argument("-m", "--model", nargs=1, required=True, help="crf model to be used", dest="model")
	return parser

if __name__ == "__main__":
	parser = argumentParser()
	arguments = parser.parse_args()

	file_name = arguments.article[0]
	model = arguments.model[0]

	with open(file_name, "r") as f:
		data = f.read()
	data = article2trainFormat(data)
	features = extract_features(data)

	predicted = model_predict(model, features)

	word_token = [i[0] for i in data]
	word_predicted = [(word, predict) for word, predict in zip(word_token, predicted)]
	predicted_word_label = gen_text_label(word_predicted)

	actors = []
	malwares = []
	campaigns = []
	targets = []

	for word, label in predicted_word_label:
		if label == "Actor":
			actors.append(word)
		elif label == "Malware":
			malwares.append(word)
		elif label == "Campaign":
			campaigns.append(word)
		elif label == "Target":
			targets.append(word)

	# Find the most occured words in each category
	actors = list(Counter(actors))
	malwares = list(Counter(malwares))
	campaigns = list(Counter(campaigns))
	targets = list(Counter(targets))

	print("Recognized entities:\n")
	print("Actors: " , actors)
	print("Malwares: ", malwares)
	print("Campaigns: ", campaigns)
	print("Targets: ", targets)

	if not actors:
		print("No actor")
	else:
		graph = Graph()
		graph.parse("tagraph.ttl", format="turtle")
		tgo_uri = "http://example.org/cyber/tgo#"

		actors_ = [i.lower().strip().replace(" ", "_") for i in actors]
		malwares_ = [i.lower().strip().replace(" ", "_") for i in malwares]
		campaigns_ = [i.lower().strip().replace(" ", "_") for i in campaigns]
		targets_ = [i.lower().strip().replace(" ", "_" ) for i in targets]

		# add all named entities into respective class
		for i in range(len(actors_)):
			graph.add((URIRef(tgo_uri + actors_[i]), RDF.type, URIRef(tgo_uri + "ThreatActor")))
			graph.add((URIRef(tgo_uri + actors_[i]), URIRef(tgo_uri + "hasTitle"), Literal(actors[i], datatype=XSD.string)))
		for i in range(len(malwares_)):
			graph.add((URIRef(tgo_uri + malwares_[i]), RDF.type, URIRef(tgo_uri + "Malware")))
			graph.add((URIRef(tgo_uri + malwares_[i]), URIRef(tgo_uri + "hasTitle"), Literal(malwares[i], datatype=XSD.string)))
		for i in range(len(campaigns_)):
			graph.add((URIRef(tgo_uri + campaigns_[i]), RDF.type, URIRef(tgo_uri + "Campaign")))
			graph.add((URIRef(tgo_uri + campaigns_[i]), URIRef(tgo_uri + "hasTitle"), Literal(campaigns[i], datatype=XSD.string)))
		for i in range(len(targets_)):
			graph.add((URIRef(tgo_uri + targets_[i]), RDF.type, URIRef(tgo_uri + "TargetField")))
			graph.add((URIRef(tgo_uri + targets_[i]), URIRef(tgo_uri + "hasTitle"), Literal(targets[i], datatype=XSD.string)))

		main_actor = actors_[0]
		if len(actors_) > 1:
			for actor in actors_[1:]:
				graph.add((URIRef(tgo_uri + main_actor), URIRef(tgo_uri + "hasAlias"), URIRef(tgo_uri + actor)))
		for malware in malwares_:
			graph.add((URIRef(tgo_uri + main_actor), URIRef(tgo_uri + "hasAssociatedMalware"), URIRef(tgo_uri + malware)))
			graph.add((URIRef(tgo_uri + malware), URIRef(tgo_uri + "isUsedBy"), URIRef(tgo_uri + main_actor)))
		for campaign in campaigns_:
			graph.add((URIRef(tgo_uri + main_actor), URIRef(tgo_uri + "hasAssociatedCampaign"), URIRef(tgo_uri + campaign)))
			graph.add((URIRef(tgo_uri + campaign), URIRef(tgo_uri + "isLaunchedBy"), URIRef(tgo_uri + main_actor)))
		for target in targets_:
			graph.add((URIRef(tgo_uri + main_actor), URIRef(tgo_uri + "hasTargetedField"), URIRef(tgo_uri + target)))
			graph.add((URIRef(tgo_uri + target), URIRef(tgo_uri + "isFocusedBy"), URIRef(tgo_uri + main_actor)))

		# save the graph
		with open(file_name.split(".")[0] + ".ttl", "wb") as f:
			f.write(graph.serialize(format="turtle"))
		print("Created " + file_name.split(".")[0] + ".ttl")