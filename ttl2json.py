import argparse
import rdflib
import json

def argumentParser():
	parser = argparse.ArgumentParser(description="Train NER using CRF algorithms")
	parser.add_argument("-t", "--article-folder", nargs=1, help="TTL file to be converted into JSON file", dest="ttl")
	return parser

if __name__ == "__main__":
	parser = argumentParser()
	arguments = parser.parse_args()

	file_name = arguments.ttl[0]
	
	graph = rdflib.Graph()
	graph.parse(file_name, format="turtle")

	namespace = list(graph.namespaces())
	name = [i[0] for i in namespace]
	ref = list(map(str, [i[1] for i in namespace]))
	ref = [i.replace("#", "") for i in ref]
	ref_name_map = {}
	for n, r in zip(name, ref):
		ref_name_map[r] = n

	nodes = []
	color_nodes = {}
	edges = []
	for s, p, o in graph:
		s = s.split("#")
		p = p.split("#")
		o = o.split("#")
		nodes.append(ref_name_map[s[0]] + "." + s[1])
		if p[1] == "type":
			# change node color here
			entities = {"ThreatActor": "red", "Malware": "orange", "Campaign": "yello", "TargetField": "green"}
			if o[1] in entities.keys():
				color_nodes[ref_name_map[s[0]] + "." + s[1]] = entities[o[1]]
		if p[1] == "hasTitle":
			nodes.append(o[0])
			edges.append((
				ref_name_map[s[0]] + "." + s[1],
				ref_name_map[p[0]] + "." + p[1], 
				o[0]
			))
		else:
			nodes.append(ref_name_map[o[0]] + "." + o[1])
			edges.append((
				ref_name_map[s[0]] + "." + s[1], 
				ref_name_map[p[0]] + "." + p[1], 
				ref_name_map[o[0]] + "." + o[1]
			))
	
	nodes = list(set(nodes))

	data = {}
	data["nodes"] = []
	for node in nodes:
		if node in color_nodes.keys():
			data["nodes"].append(
				{
					"id": node,
					"label": node,
					"size": 1,
					"color": color_nodes[node]
				}
			)
		else:
			data["nodes"].append(
				{
					"id": node,
					"label": node,
					"size": 1,
					# "color": "red"
				}
			)
		
	data["edges"] = []
	edge_count = 0
	for edge in edges:
		data["edges"].append(
			{
				"id": "e" + str(edge_count),
				"source": edge[0],
				"target": edge[2],
				"label": edge[1],
				"type": "curvedArrow"
			}
		)
		edge_count += 1
	
	with open(file_name.split(".")[0] + ".json", "w") as outfile:
		json.dump(data, outfile)