import rdflib


# Claims
def get_number_statements(graph: rdflib.Graph):
    ans = graph.query('SELECT DISTINCT ?statement WHERE {?statement a gaf:Assertion .}')
    return len(ans)


def get_number_grasp_instances(graph: rdflib.Graph):
    ans = graph.query('SELECT DISTINCT ?instance WHERE {?instance a gaf:Instance .}')
    return len(ans)


# Perspectives
def get_number_perspectives(graph: rdflib.Graph):
    ans = graph.query('SELECT DISTINCT ?perspective WHERE {?perspective a grasp:Attribution .}')
    return len(ans)


def get_number_mentions(graph: rdflib.Graph):
    ans = graph.query('SELECT DISTINCT ?mention WHERE {?mention a gaf:Mention .}')
    return len(ans)


# Interactions
def get_number_chats(graph: rdflib.Graph):
    ans = graph.query('SELECT DISTINCT ?chat WHERE {?chat a gaf:Chat .}')
    return len(ans)


def get_number_utterances(graph: rdflib.Graph):
    ans = graph.query('SELECT DISTINCT ?utt WHERE {?utt a grasp:Utterance .}')
    return len(ans)


def get_number_sources(graph: rdflib.Graph):
    ans = graph.query('SELECT DISTINCT ?source WHERE {?source a grasp:Source .}')
    return len(ans)

# Conflicts

# Gaps


# Ratios
# mentions per instance? per claim? -> same things mentioned many times
# per utterance ->
# per attribution? ->
# attributions per source? -> knowledge per source
