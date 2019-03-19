from rdflib import Graph, Namespace
from rdflib.namespace import RDF, XSD, RDFS
from rdflib.term import Literal

graph = Graph()
tgo = Namespace("http://example.org/cyber/tgo#")
graph.bind("tgo", tgo)

graph.add((tgo.ThreatActor, RDF.type, RDFS.Class))
graph.add((tgo.Malware, RDF.type, RDFS.Class))
graph.add((tgo.Campaign, RDF.type, RDFS.Class))
graph.add((tgo.TargetField, RDF.type, RDFS.Class))

graph.add((tgo.hasTitle, RDFS.domain, tgo.ThreatActor))
graph.add((tgo.hasTitle, RDFS.domain, tgo.Campaign))
graph.add((tgo.hasTitle, RDFS.domain, tgo.Malware))
graph.add((tgo.hasTitle, RDFS.domain, tgo.TargetField))
graph.add((tgo.hasTitle, RDFS.range, XSD.string))

graph.add((tgo.hasAssociatedCampaign, RDFS.domain, tgo.ThreatActor))
graph.add((tgo.hasAssociatedCampaign, RDFS.range, tgo.Campaign))
graph.add((tgo.hasAssociatedMalware, RDFS.domain, tgo.ThreatActor))
graph.add((tgo.hasAssociatedMalware, RDFS.range, tgo.Malware))
graph.add((tgo.hasTargetedField, RDFS.domain, tgo.ThreatActor))
graph.add((tgo.hasTargetedField, RDFS.range, tgo.TargetField))
graph.add((tgo.hasAlias, RDFS.domain, tgo.ThreatActor))
graph.add((tgo.hasAlias, RDFS.range, tgo.ThreatActor))

graph.add((tgo.isUsedBy, RDFS.domain, tgo.Malware))
graph.add((tgo.isUsedBy, RDFS.range, tgo.ThreatActor))

graph.add((tgo.isLaunchedBy, RDFS.domain, tgo.Campaign))
graph.add((tgo.isLaunchedBy, RDFS.range, tgo.ThreatActor))

graph.add((tgo.isFocusedBy, RDFS.domain, tgo.TargetField))
graph.add((tgo.isFocusedBy, RDFS.range, tgo.ThreatActor))

# save the graph
with open("tagraph.ttl", "wb") as f:
    f.write(graph.serialize(format="turtle"))