import os
from datetime import date
from pathlib import Path
from random import getrandbits
from tempfile import TemporaryDirectory

import requests
from rdflib import Namespace

ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))
RESOURCES_PATH = ABSOLUTE_PATH + "/../../../resources/"

############################################################
BRAIN_ADDRESS = "http://localhost:7200/repositories/sandbox"
CONTEXT_ID = getrandbits(8)
PLACE_ID = getrandbits(8)
PLACE_NAME = "office"
LOCATION = requests.get("https://ipinfo.io").json()

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
    "context_id": CONTEXT_ID,
    "date": date.today()
}

############################################################
ONTOLOGY_DETAILS = {"filepath": "/Users/sbaez/Documents/PhD/data/harry potter dataset/Data/EN-data/ontology.ttl",
                    "namespace": "http://harrypotter.org/",
                    "prefix": "hp"}

############################################################
from cltl.brain.basic_brain import BasicBrain

with TemporaryDirectory(prefix="brain-log") as log_path:
    BASIC_BRAIN = BasicBrain(address=BRAIN_ADDRESS, log_dir=Path(log_path), ontology_details=ONTOLOGY_DETAILS,
                             clear_all=False)
PREDICATE_OPTIONS = BASIC_BRAIN.get_predicates()

HARRYPOTTER_NS = Namespace("http://harrypotter.org/")
HARRYPOTTER_PREFIX = "hp"

############################################################
import logging

from cltl.brain import logger as brain_logger

brain_logger.setLevel(logging.ERROR)

from cltl.thoughts.thought_selection import logger as thoughts_logger

thoughts_logger.setLevel(logging.ERROR)
