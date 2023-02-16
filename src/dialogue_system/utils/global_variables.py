import os
from datetime import date
from pathlib import Path
from random import getrandbits

import requests

ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))
RESOURCES_PATH = ABSOLUTE_PATH + "/../../../resources/"

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

from cltl.brain.basic_brain import BasicBrain

BASIC_BRAIN = BasicBrain("http://localhost:7200/repositories/sandbox", log_dir=Path(RESOURCES_PATH))
PREDICATE_OPTIONS = BASIC_BRAIN.get_predicates()

