import argparse
from rdflib import Graph
from rdflib.term import URIRef

def argumentParser():
	parser = argparse.ArgumentParser(description="Train NER using CRF algorithms")
	parser.add_argument("-f", "--ttl-files", nargs="+", required=True, help="TTL files to be merged", dest="ttl_files")
	return parser

if __name__ == "__main__":
	parser = argumentParser()
	arguments = parser.parse_args()

	final_graph = Graph()
	tgo = URIRef("http://example.org/cyber/tgo#")
	final_graph.bind("tgo", tgo)

	for ttl_file in arguments.ttl_files:
		temp_graph = Graph()
		temp_graph.parse(ttl_file, format="turtle")
		final_graph += temp_graph

	file_name = "graph.ttl"
	with open(file_name, "wb") as f:
		f.write(final_graph.serialize(format="turtle"))