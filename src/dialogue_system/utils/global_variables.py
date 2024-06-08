import os
from random import getrandbits

from rdflib import Namespace, URIRef

############################## HP DATA PREFIXES ##############################
HARRYPOTTER_NS = Namespace("http://harrypotter.org/")
HARRYPOTTER_PREFIX = "hp"

############################## PATHS FOR CHATBOT LEARNING ##############################
ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))
RESOURCES_PATH = ABSOLUTE_PATH + "/../../../resources/"

############################## PATHS FOR HP DATA PROCESSING ##############################
CHARACTER_TYPE_PATH = RESOURCES_PATH + "hp_data/character_types.csv"
ATTRIBUTE_TYPE_PATH = RESOURCES_PATH + "hp_data/attribute_types.csv"

OG_DATA_PATHS = [RESOURCES_PATH + "hp_data/test_set_en/", RESOURCES_PATH + "hp_data/train_set_en/"]
USER_PATH = RESOURCES_PATH + "users"
RAW_USER_PATH = USER_PATH + "/raw" # TODO when running locally to raw_test
RAW_VANILLA_USER_PATH = RAW_USER_PATH + "/vanilla.trig"
PROCESSED_USER_PATH = USER_PATH + "/processed"

############################## BRAIN DETAILS FOR CHATBOT ##############################
ONTOLOGY_DETAILS = {"filepath": RESOURCES_PATH + "hp_data/ontology.ttl",
                    "namespace": "http://harrypotter.org/",
                    "prefix": "hp"}
BRAIN_ADDRESS = "http://localhost:7200/repositories/sandbox"

############################## CAPSULE FOR CHATBOT ##############################
CONTEXT_ID = getrandbits(8)
PLACE_ID = getrandbits(8)
PLACE_NAME = "office"
# LOCATION = requests.get("https://ipinfo.io").json()
LOCATION = {"country": "MX", "region": "Mexico City", "city": "Mexico City"}
BASE_CAPSULE = {
    "chat": None,  # from dialogue_system / prev capsule
    "turn": None,  # from dialogue_system
    "author": None,  # from dialogue_system
    "utterance": "",
    "utterance_type": None,
    "position": "",
    "subject": {"label": None, "type": [], 'uri': None},
    "predicate": {"label": None, 'uri': None},
    "object": {"label": None, "type": [], 'uri': None},
    "perspective": {"certainty": None, "polarity": None, "sentiment": None},
    "context_id": CONTEXT_ID
}

############################## EKG GENERAL URIS ##############################
PERSPECTIVE_GRAPH = URIRef("http://cltl.nl/leolani/talk/Perspectives")
CLAIM_GRAPH = URIRef("http://cltl.nl/leolani/world/Claims")
INSTANCE_GRAPH = URIRef("http://cltl.nl/leolani/world/Instances")
TYPE_EVENT = URIRef("http://semanticweb.cs.vu.nl/2009/11/sem/Event")
TYPE_ASSERTION = URIRef("http://groundedannotationframework.org/gaf#Assertion")
TYPE_ATTRIBUTION = URIRef("http://groundedannotationframework.org/grasp#Attribution")
TYPE_ATTRIBUTIONVALUE = URIRef("http://groundedannotationframework.org/grasp#AttributionValue")
TYPE_CERTAINTYVALUE = URIRef("http://groundedannotationframework.org/grasp/factuality#CertaintyValue")
TYPE_POLARITYVALUE = URIRef("http://groundedannotationframework.org/grasp/factuality#PolarityValue")
TYPE_TEMPORALVALUE = URIRef("http://groundedannotationframework.org/grasp/factuality#TemporalValue")
TYPE_SENTIMENTVALUE = URIRef("http://groundedannotationframework.org/grasp/sentiment#SentimentValue")
TYPE_EMOTIONVALUE = URIRef("http://groundedannotationframework.org/grasp/emotion#EmotionValue")
CERTAINTY_CERTAIN = URIRef("http://groundedannotationframework.org/grasp/factuality#CERTAIN")
CERTAINTY_PROBABLE = URIRef("http://groundedannotationframework.org/grasp/factuality#PROBABLE")
CERTAINTY_POSSIBLE = URIRef("http://groundedannotationframework.org/grasp/factuality#POSSIBLE")
CERTAINTY_UNDERSPECIFIED = URIRef("http://groundedannotationframework.org/grasp/factuality#UNDERSPECIFIED")
POLARITY_POSITIVE = URIRef("http://groundedannotationframework.org/grasp/factuality#POSITIVE")
POLARITY_NEGATIVE = URIRef("http://groundedannotationframework.org/grasp/factuality#NEGATIVE")
TEMPORALITY_NONFUTURE = URIRef("http://groundedannotationframework.org/grasp/factuality#NONFUTURE")
TEMPORALITY_FUTURE = URIRef("http://groundedannotationframework.org/grasp/factuality#FUTURE")
SENTIMENT_POSITIVE = URIRef("http://groundedannotationframework.org/grasp/sentiment#POSITIVE")
SENTIMENT_NEGATIVE = URIRef("http://groundedannotationframework.org/grasp/sentiment#NEGATIVE")
SENTIMENT_NEUTRAL = URIRef("http://groundedannotationframework.org/grasp/sentiment#NEUTRAL")
SENTIMENT_UNDERSPECIFIED = URIRef("http://groundedannotationframework.org/grasp/sentiment#UNDERSPECIFIED")
EMOTION_ANGER = URIRef("http://groundedannotationframework.org/grasp/emotion#ANGER")
EMOTION_DISGUST = URIRef("http://groundedannotationframework.org/grasp/emotion#DISGUST")
EMOTION_FEAR = URIRef("http://groundedannotationframework.org/grasp/emotion#FEAR")
EMOTION_HAPPINESS = URIRef("http://groundedannotationframework.org/grasp/emotion#HAPPINESS")
EMOTION_SADNESS = URIRef("http://groundedannotationframework.org/grasp/emotion#SADNESS")
EMOTION_SURPRISE = URIRef("http://groundedannotationframework.org/grasp/emotion#SURPRISE")
EMOTION_NEUTRAL = URIRef("http://groundedannotationframework.org/grasp/emotion#NEUTRAL")
EMOTION_UNDERSPECIFIED = URIRef("http://groundedannotationframework.org/grasp/emotion#UNDERSPECIFIED")
GRASP_HASATT = URIRef("http://groundedannotationframework.org/grasp#hasAttribution")
GRASP_ATTFOR = URIRef("http://groundedannotationframework.org/grasp#isAttributionFor")
GAF_DENOTEDBY = URIRef("http://groundedannotationframework.org/gaf#denotedBy")
GAF_DENOTES = URIRef("http://groundedannotationframework.org/gaf#denotes")
GAF_DENOTEDIN = URIRef("http://groundedannotationframework.org/gaf#denotedIn")
GAF_CONTAINSDEN = URIRef("http://groundedannotationframework.org/gaf#containsDenotation")
SIMPLE_ATTRELS = [TYPE_CERTAINTYVALUE, TYPE_POLARITYVALUE, TYPE_TEMPORALVALUE,
                  TYPE_SENTIMENTVALUE,
                  TYPE_EMOTIONVALUE]
SIMPLE_ATTVALS = [CERTAINTY_CERTAIN, CERTAINTY_PROBABLE, CERTAINTY_POSSIBLE, CERTAINTY_UNDERSPECIFIED,
                  POLARITY_POSITIVE, POLARITY_NEGATIVE,
                  TEMPORALITY_NONFUTURE, TEMPORALITY_FUTURE,
                  SENTIMENT_POSITIVE, SENTIMENT_NEGATIVE, SENTIMENT_NEUTRAL, SENTIMENT_UNDERSPECIFIED,
                  EMOTION_ANGER, EMOTION_DISGUST, EMOTION_FEAR, EMOTION_HAPPINESS, EMOTION_SADNESS, EMOTION_SURPRISE,
                  EMOTION_NEUTRAL, EMOTION_UNDERSPECIFIED]
ATTVALS_TO_ATTRELS = {CERTAINTY_CERTAIN: TYPE_CERTAINTYVALUE, CERTAINTY_PROBABLE: TYPE_CERTAINTYVALUE,
                      CERTAINTY_POSSIBLE: TYPE_CERTAINTYVALUE, CERTAINTY_UNDERSPECIFIED: TYPE_CERTAINTYVALUE,

                      POLARITY_POSITIVE: TYPE_POLARITYVALUE, POLARITY_NEGATIVE: TYPE_POLARITYVALUE,

                      TEMPORALITY_NONFUTURE: TYPE_TEMPORALVALUE, TEMPORALITY_FUTURE: TYPE_TEMPORALVALUE,

                      SENTIMENT_POSITIVE: TYPE_SENTIMENTVALUE, SENTIMENT_NEGATIVE: TYPE_SENTIMENTVALUE,
                      SENTIMENT_NEUTRAL: TYPE_SENTIMENTVALUE, SENTIMENT_UNDERSPECIFIED: TYPE_SENTIMENTVALUE,

                      EMOTION_ANGER: TYPE_EMOTIONVALUE, EMOTION_DISGUST: TYPE_EMOTIONVALUE,
                      EMOTION_FEAR: TYPE_EMOTIONVALUE, EMOTION_HAPPINESS: TYPE_EMOTIONVALUE,
                      EMOTION_SADNESS: TYPE_EMOTIONVALUE, EMOTION_SURPRISE: TYPE_EMOTIONVALUE,
                      EMOTION_NEUTRAL: TYPE_EMOTIONVALUE, EMOTION_UNDERSPECIFIED: TYPE_EMOTIONVALUE
                      }
