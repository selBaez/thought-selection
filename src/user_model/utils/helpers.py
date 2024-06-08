from pathlib import Path
from random import getrandbits, sample

from rdflib import URIRef, Literal, RDF, RDFS

from cltl.brain.utils.helper_functions import hash_claim_id
from dialogue_system.utils.global_variables import OG_DATA_PATHS, \
    PERSPECTIVE_GRAPH, CLAIM_GRAPH, INSTANCE_GRAPH, TYPE_EVENT, TYPE_ASSERTION, TYPE_ATTRIBUTION, TYPE_ATTRIBUTIONVALUE, \
    TYPE_CERTAINTYVALUE, TYPE_POLARITYVALUE, CERTAINTY_CERTAIN, CERTAINTY_POSSIBLE, POLARITY_POSITIVE, \
    POLARITY_NEGATIVE, SENTIMENT_NEUTRAL, EMOTION_UNDERSPECIFIED, GRASP_HASATT, GRASP_ATTFOR, GAF_DENOTEDBY, \
    GAF_DENOTES, GAF_DENOTEDIN, GAF_CONTAINSDEN
from dialogue_system.utils.helpers import build_graph


def break_list(text):
    """
    Split elements in a list
    """
    for sep in [",", " and "]:
        text = text.replace(sep, ",")

    text = text.split(",")

    return text


def clean_string(text):
    """
    Get a string and remove special characters and empty spaces.
    Casefolding to first letter capitalized (arbitrary decision)
    Remove conjunctions/articles at the beginning
    """
    # Remove
    for punct in ["(", ")", ".", "'", "!", "，", "’"]:
        text = text.replace(punct, "")

    # Replace with spaces
    for punct in ["-"]:
        text = text.replace(punct, " ")

    text = text.replace("  ", " ")
    text = text.strip()
    text = text.lower()

    # POS tagging
    # words = word_tokenize(text)
    # poss = pos_tag(words)
    # if poss[0] and (poss[0][1] in ["CC", "DT"]):
    # #     # remove first word. Think? do we need to remove all conjuctions or just the "and"
    #     pass

    if text.startswith("And "):
        text = text[4:]
        text = text.lower()

    # Future work: if string is longer than 4 tokens remove it

    text = text.replace(" ", "-")
    return text.lower()


def get_book_number(file):
    """
    Get book number from the file name
    """
    book = file.stem.split('_')[0][4:]
    return book


def get_all_files(extension="json"):
    """
    Get all JSON files that belong to the dataset
    """
    files = []
    for folder in OG_DATA_PATHS:
        files.extend(list(Path(folder).glob(f'*.{extension}')))

    files.sort()

    return files


def merge_all_graphs(test=False):
    """
    Put together triples from different trig files
    """
    graph_data = build_graph()
    trig_files = get_all_files(extension="trig")
    for trig_file in trig_files:
        graph_data.parse(trig_file)
        print(f"READ DATASET in {trig_file.stem}: {len(graph_data)}")
        if test:
            break

    return graph_data


def save_graph(filepath, graph_data):
    """
    Write graph to trig file
    """
    graph_data.serialize(destination=filepath, format="trig")
    print(f"\tFINAL SIZE OF DATASET in {filepath.stem}: {len(graph_data)}")


def sample_character(all_characters):
    """
    Get a random node of type character
    """
    selected_character = sample(all_characters, 1)

    return selected_character[0]["character"]


def sample_attribute(all_attributes):
    """
    Get a random node of type attribute
    """
    selected_attribute = sample(all_attributes, 1)

    return selected_attribute[0]["attribute"]


def get_all_attribution_values(graph_data):
    """
    Query graph for attribution values (perspectives)
    """
    q_att_vals = """SELECT distinct ?attributionVal  WHERE {{ ?attributionVal rdf:type grasp:AttributionValue . }}"""
    all_att_vals = graph_data.query(q_att_vals)
    all_att_vals = [c for c in all_att_vals]

    print(f"ATTRIBUTION VALUES IN DATASET: {len(all_att_vals)}")

    return all_att_vals


def get_all_claims(graph_data):
    """
    Query graph for claims
    """
    q_claims = """SELECT distinct ?claim  WHERE {{ 
                  ?claim rdf:type gaf:Assertion . ?claim gaf:denotedBy ?mention .}}"""
    all_claims = graph_data.query(q_claims)
    all_claims = [c for c in all_claims]

    print(f"CLAIMS IN DATASET: {len(all_claims)}")

    return all_claims


def sample_claims(all_claims):
    """
    Select half of the claims, randomly
    """
    sample_size = int(len(all_claims) / 2)
    claims_to_delete = sample(all_claims, sample_size)

    return claims_to_delete


def get_claim_elements(graph_data, claim):
    """
    Query graph for triple in this claim
    """
    q_elems = """SELECT distinct ?s ?p ?o  WHERE {{ 
                GRAPH ?claim {{?s ?p ?o .}}
                }}"""
    all_elems = graph_data.query(q_elems, initBindings={'claim': claim})
    all_elems = [a for a in all_elems]

    elems = all_elems[0]  # we only have a triple per claim

    return elems


def get_all_mentions_of_claim(graph_data, claim):
    """
    Query graph for mentions of claims
    """
    q_men = """SELECT distinct ?mention  WHERE {{ ?claim gaf:denotedBy ?mention }}"""
    all_men = graph_data.query(q_men, initBindings={'claim': claim})
    all_men = [m for m in all_men]

    return all_men


def get_all_attribution_of_claim(graph_data, claim):
    """
    Query graph for attributions of this claim
    """
    q_atts = """SELECT distinct ?att  WHERE {{ 
                ?att grasp:isAttributionFor ?mention .
                ?claim gaf:denotedBy ?mention }}"""
    all_atts = graph_data.query(q_atts, initBindings={'claim': claim})
    all_atts = [a for a in all_atts]

    if len(all_atts) > 0:
        att = all_atts[0]["att"]  # we only have an att per claim since we had the vanilla user as base
    else:
        att = ""

    return att


def link_mention_to_att(graph_data, men, att):
    """
    Add links between new attribution and mention
    """
    men = men["mention"]
    graph_data.add((men, GRASP_HASATT, att, PERSPECTIVE_GRAPH))
    graph_data.add((att, GRASP_ATTFOR, men, PERSPECTIVE_GRAPH))

    return graph_data


def link_mention_to_claim(graph_data, men, claim, elems):
    """
    Establish gaf relations between mentions and claims
    """
    graph_data.add((claim, GAF_DENOTEDBY, men["mention"], CLAIM_GRAPH))
    graph_data.add((men["mention"], GAF_DENOTES, claim, PERSPECTIVE_GRAPH))
    graph_data.add((men["mention"], GAF_CONTAINSDEN, elems["s"], PERSPECTIVE_GRAPH))
    graph_data.add((men["mention"], GAF_CONTAINSDEN, elems["o"], PERSPECTIVE_GRAPH))
    graph_data.add((elems["s"], GAF_DENOTEDIN, men["mention"], INSTANCE_GRAPH))
    graph_data.add((elems["o"], GAF_DENOTEDIN, men["mention"], INSTANCE_GRAPH))

    return graph_data


def create_low_certainty(u_graph):
    """
    Add probable certainty
    """
    u_graph.add((CERTAINTY_POSSIBLE, RDF.type, TYPE_ATTRIBUTIONVALUE, PERSPECTIVE_GRAPH))
    u_graph.add((CERTAINTY_POSSIBLE, RDF.type, TYPE_CERTAINTYVALUE, PERSPECTIVE_GRAPH))
    u_graph.add((CERTAINTY_POSSIBLE, RDFS.label, Literal("PROBABLE"), PERSPECTIVE_GRAPH))

    return u_graph


def create_neg_polairty(u_graph):
    """
    Add negative polarity
    """
    u_graph.add((POLARITY_NEGATIVE, RDF.type, TYPE_ATTRIBUTIONVALUE, PERSPECTIVE_GRAPH))
    u_graph.add((POLARITY_NEGATIVE, RDF.type, TYPE_POLARITYVALUE, PERSPECTIVE_GRAPH))
    u_graph.add((POLARITY_NEGATIVE, RDFS.label, Literal("NEGATIVE"), PERSPECTIVE_GRAPH))

    return u_graph


def lower_certainty(graph_data, att):
    """
    Make new attribution, with certainty "possible"
    """
    new_label = att[:-4] + "1120"
    new_att = URIRef(new_label)
    new_label = Literal(new_label.split("/")[-1])

    graph_data.add((new_att, RDF.type, TYPE_ATTRIBUTION, PERSPECTIVE_GRAPH))
    graph_data.add((new_att, RDFS.label, new_label, PERSPECTIVE_GRAPH))
    graph_data.add((new_att, RDF.value, EMOTION_UNDERSPECIFIED, PERSPECTIVE_GRAPH))
    graph_data.add((new_att, RDF.value, CERTAINTY_POSSIBLE, PERSPECTIVE_GRAPH))
    graph_data.add((new_att, RDF.value, POLARITY_POSITIVE, PERSPECTIVE_GRAPH))
    graph_data.add((new_att, RDF.value, SENTIMENT_NEUTRAL, PERSPECTIVE_GRAPH))

    return graph_data, new_att


def negate_polarity(graph_data, att):
    """
    Make new attribution, with negative polarity
    """
    new_label = att[:-4] + "3-120"
    new_att = URIRef(new_label)
    new_label = Literal(new_label.split("/")[-1])

    graph_data.add((new_att, RDF.type, TYPE_ATTRIBUTION, PERSPECTIVE_GRAPH))
    graph_data.add((new_att, RDFS.label, new_label, PERSPECTIVE_GRAPH))
    graph_data.add((new_att, RDF.value, EMOTION_UNDERSPECIFIED, PERSPECTIVE_GRAPH))
    graph_data.add((new_att, RDF.value, CERTAINTY_CERTAIN, PERSPECTIVE_GRAPH))
    graph_data.add((new_att, RDF.value, POLARITY_NEGATIVE, PERSPECTIVE_GRAPH))
    graph_data.add((new_att, RDF.value, SENTIMENT_NEUTRAL, PERSPECTIVE_GRAPH))

    return graph_data, new_att


def create_new_claim(graph_data, claim, all_characters, all_attributes):
    """
    Make new claim with subject or object switched (valid node type checked)
    """
    elems = get_claim_elements(graph_data, claim)

    # Select what to swap
    swap_subject = bool(getrandbits(1))

    if swap_subject:
        new_s = sample_character(all_characters)
        claim_label = hash_claim_id([new_s.split('/')[-1], elems["p"].split('/')[-1], elems["o"].split('/')[-1]])
        new_elems = {"s": new_s, "p": elems["p"], "o": elems["o"]}

    else:
        new_o = sample_attribute(all_attributes)
        claim_label = hash_claim_id([elems["s"].split('/')[-1], elems["p"].split('/')[-1], new_o.split('/')[-1]])
        new_elems = {"s": elems["s"], "p": elems["p"], "o": new_o}

    new_claim = URIRef("http://cltl.nl/leolani/world/" + claim_label)
    graph_data.add((new_claim, RDF.type, TYPE_ASSERTION, CLAIM_GRAPH))
    graph_data.add((new_claim, RDF.type, TYPE_EVENT, CLAIM_GRAPH))
    graph_data.add((new_claim, RDFS.label, Literal(claim_label), CLAIM_GRAPH))

    return new_claim, elems, new_elems


def create_new_att(graph_data, claim):
    """
    Make new attribution, with basic perspective values
    """
    new_label = claim + "_3120"
    new_att = URIRef(new_label)
    new_label = Literal(new_label.split("/")[-1])

    graph_data.add((new_att, RDF.type, TYPE_ATTRIBUTION, PERSPECTIVE_GRAPH))
    graph_data.add((new_att, RDFS.label, new_label, PERSPECTIVE_GRAPH))
    graph_data.add((new_att, RDF.value, EMOTION_UNDERSPECIFIED, PERSPECTIVE_GRAPH))
    graph_data.add((new_att, RDF.value, CERTAINTY_CERTAIN, PERSPECTIVE_GRAPH))
    graph_data.add((new_att, RDF.value, POLARITY_POSITIVE, PERSPECTIVE_GRAPH))
    graph_data.add((new_att, RDF.value, SENTIMENT_NEUTRAL, PERSPECTIVE_GRAPH))

    return graph_data, new_att
