from datetime import date
from random import getrandbits

import requests

CONTEXT_ID = getrandbits(8)
PLACE_ID = getrandbits(8)
PLACE_NAME = "office"
LOCATION = requests.get("https://ipinfo.io").json()

BASE_CAPSULE = {
    "chat": None,  # from chatbot / prev capsule
    "turn": None,  # from chatbot
    "author": None,  # from chatbot
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
