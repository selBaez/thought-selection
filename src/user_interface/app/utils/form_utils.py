from datetime import datetime

from src.dialogue_system.utils.global_variables import CONTEXT_ID, ONTOLOGY_DETAILS


def statement_capsule_to_form(capsule, form):
    # Triple
    form.subject_label.data = capsule["subject"]["label"]
    form.subject_types.data = ','.join(capsule["subject"]["type"])
    form.subject_uri.data = f'http://cltl.nl/leolani/world/{capsule["subject"]["label"]}'
    form.predicate_label.data = capsule["predicate"]["label"]
    form.predicate_uri.data = f'{ONTOLOGY_DETAILS["namespace"]}{capsule["predicate"]["label"]}'
    form.object_label.data = capsule["object"]["label"]
    form.object_types.data = ','.join(capsule["object"]["type"])
    form.object_uri.data = f'http://cltl.nl/leolani/world/{capsule["object"]["label"]}'

    # perspective
    form.perspective_certainty.data = capsule["perspective"]["certainty"]
    form.perspective_polarity.data = capsule["perspective"]["polarity"]
    form.perspective_sentiment.data = capsule["perspective"]["sentiment"]

    # chat
    form.turn_id.data = capsule["turn"]

    # utterance
    form.utterance.data = capsule["utterance"]
    form.utterance_type.data = capsule["utterance_type"]
    # form.position.data = capsule["position"]

    return form


def form_to_context_capsule(form):
    capsule = {}

    # context
    capsule["context_id"] = form.context_id.data
    capsule["date"] = datetime.strftime(form.context_date.data, "%Y-%m-%d")

    # place
    capsule["place"] = form.place_label.data
    capsule["place_id"] = form.place_id.data
    capsule["country"] = form.country.data
    capsule["region"] = form.region.data
    capsule["city"] = form.city.data

    return capsule


def form_to_statement_capsule(form, chatbot):
    capsule = {}

    capsule["chat"] = chatbot.chat_id
    capsule["turn"] = form.turn_id.data
    capsule["author"] = {"label": chatbot.speaker.lower(),
                         "type": ["person"],
                         "uri": f"http://cltl.nl/leolani/world/{chatbot.speaker.lower()}"}
    capsule["utterance"] = form.utterance.data
    capsule["utterance_type"] = form.utterance_type.data
    capsule["position"] = ""

    capsule["subject"] = {"label": form.subject_label.data.lower(),
                          "type": form.subject_types.data.split(','),
                          "uri": f"http://cltl.nl/leolani/world/{form.subject_label.data.lower()}"}
    capsule["predicate"] = {"label": form.predicate_label.data,
                            "uri": f"{ONTOLOGY_DETAILS['namespace']}{form.predicate_label.data.lower()}"}
    capsule["object"] = {"label": form.object_label.data.lower(),
                         "type": form.object_types.data.split(','),
                         "uri": f"http://cltl.nl/leolani/world/{form.object_label.data.lower()}"}
    capsule["perspective"] = {"certainty": form.perspective_certainty.data,
                              "polarity": form.perspective_polarity.data,
                              "sentiment": form.perspective_sentiment.data}

    # context
    capsule["context_id"] = CONTEXT_ID
    capsule["timestamp"] = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S.%f")

    return capsule


def digest_form(form, chatbot):
    # turn form into capsule, so it can be picked up by the brain
    capsule = form_to_statement_capsule(form, chatbot)

    # process with brain, get template for response
    say, capsule_user, brain_response = chatbot.respond(capsule)

    # use capsule user to fill in next form
    form = statement_capsule_to_form(capsule_user, form)
    reply = {'say': say}

    return form, reply, capsule, capsule_user


def begin_form(form, chatbot):
    capsule = {
        "chat": chatbot.chat_id,
        "turn": chatbot.turns,
        "author": {
            "label": chatbot.speaker.lower(),
            "type": ["person"],
            "uri": f"http://cltl.nl/leolani/world/{chatbot.speaker.lower()}"
        },
        "utterance": "",
        "utterance_type": "STATEMENT",
        "position": "",
        "subject": {
            "label": chatbot.speaker.lower(),
            "type": ["person"],
            "uri": f"http://cltl.nl/leolani/world/{chatbot.speaker.lower()}"
        },
        "predicate": {
            "label": "know",
            "uri": f"{ONTOLOGY_DETAILS['namespace']}know"
        },
        "object": {
            "label": "leolani",
            "type": ["robot"],
            "uri": f"http://cltl.nl/leolani/world/leolani"
        },
        "perspective": {
            "certainty": 1,
            "polarity": 1,
            "sentiment": 1
        }
    }

    # use capsule user to fill in next form
    form = statement_capsule_to_form(capsule, form)
    reply = {'say': chatbot.greet}

    # arrange all response info to be saved
    chatbot.chat_history['capsules_suggested'].append(capsule)

    return form, reply, capsule
