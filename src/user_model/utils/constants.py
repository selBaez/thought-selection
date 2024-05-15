from datetime import date
from random import getrandbits

############################################################
CONTEXT_ID = getrandbits(8)
START_DATE = date(1997, 6, 26)
HP_CONTEXT_CAPSULE = {"context_id": CONTEXT_ID,
                      "date": START_DATE,
                      "place": "Harry Potter World",
                      "place_id": 1,
                      "country": "UK",
                      "region": "Scotland",
                      "city": "Edinburgh"}

