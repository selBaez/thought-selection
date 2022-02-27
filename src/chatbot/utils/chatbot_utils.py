""" Filename:     chatbot_utils.py
    Author(s):    Thomas Bellucci
    Description:  Utility functions used by the Chatbot in Chatbot.py.
    Date created: Nov. 11th, 2021
"""


def capsule_for_query(capsule):
    """Casefolds the triple stored in a capsule so that entities match
    with the brain regardless of case.

    params
    dict capsule: capsule containing a triple

    returns: capsule with casefolded triple
    """
    for item in ["subject", "predicate", "object"]:
        if capsule[item]["label"]:
            capsule[item]["label"] = capsule[item]["label"].lower()
    return capsule
