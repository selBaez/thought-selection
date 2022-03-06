from datetime import datetime
from random import getrandbits

import requests
from cltl.brain.utils.helper_functions import brain_response_to_json

context_id = getrandbits(8)
place_id = getrandbits(8)
location = requests.get("https://ipinfo.io").json()


def capsule_to_form(capsule, form):
    # Triple
    form.subject_label.data = capsule["subject"]["label"]
    form.subject_types.data = ','.join(capsule["subject"]["type"])
    form.predicate_label.data = capsule["predicate"]["label"]
    form.object_label.data = capsule["object"]["label"]
    form.object_types.data = ','.join(capsule["object"]["type"])

    # perspective
    form.perspective_certainty.data = capsule["perspective"]["certainty"]
    form.perspective_polarity.data = capsule["perspective"]["polarity"]
    form.perspective_sentiment.data = capsule["perspective"]["sentiment"]

    # chat
    form.chat_id.data = capsule["chat"]
    form.turn_id.data = capsule["turn"]
    form.author.data = capsule["author"]

    # utterance
    form.utterance.data = capsule["utterance"]
    form.utterance_type.data = capsule["utterance_type"]
    # form.position.data = capsule["position"]

    # context
    form.context_id.data = capsule["context_id"]
    # form.context_date.data = capsule["date"]# datetime.strptime(capsule["date"], "%Y-%m-%d")

    # place
    form.place_label.data = capsule["place"]
    form.place_id.data = capsule["place_id"]
    form.country.data = capsule["country"]
    form.region.data = capsule["region"]
    form.city.data = capsule["city"]

    # multimodal
    # form.objects.data = ','.join(capsule["objects"])
    # form.people.data = '.'.join(capsule["people"])

    return form


def form_to_capsule(form, chatbot):
    capsule = {}
    capsule["chat"] = chatbot.chat_id
    capsule["turn"] = chatbot.turns
    capsule["author"] = chatbot.speaker.lower()
    capsule["utterance"] = form.utterance.data
    capsule["utterance_type"] = form.utterance_type.data
    capsule["position"] = ""

    capsule["subject"] = {"label": form.subject_label.data.lower(),
                          "type": form.subject_types.data.split(','),
                          "uri": f"http://cltl.nl/leolani/world/{form.subject_label.data.lower()}"}
    capsule["predicate"] = {"label": form.predicate_label.data,
                            "uri": f"http://cltl.nl/leolani/n2mu/{form.predicate_label.data.lower()}"}
    capsule["object"] = {"label": form.object_label.data.lower(),
                         "type": form.object_types.data.split(','),
                         "uri": f"http://cltl.nl/leolani/world/{form.object_label.data.lower()}"}
    capsule["perspective"] = {"certainty": form.perspective_certainty.data,
                              "polarity": form.perspective_polarity.data,
                              "sentiment": form.perspective_sentiment.data}

    capsule["context_id"] = form.context_id.data
    capsule["date"] = datetime.strftime(form.context_date.data, "%Y-%m-%d")
    capsule["place"] = form.place_label.data
    capsule["place_id"] = form.place_id.data
    capsule["country"] = form.country.data
    capsule["region"] = form.region.data
    capsule["city"] = form.city.data
    # capsule["objects"] = form.objects.data.split(',')
    # capsule["people"] = form.people.data.split(',')
    capsule["objects"] = []
    capsule["people"] = []

    return capsule


def digest_form(form, chatbot):
    # turn form into capsule so it can be picked up by the brain
    capsule = form_to_capsule(form, chatbot)

    # process with brain
    say, capsule_user, brain_response = chatbot.respond(capsule)

    # use capsule user to fill in next form
    form = capsule_to_form(capsule_user, form)
    reply = {'say': say}

    # arrange all response info to be saved
    capsule["brain_state"] = chatbot.replier.brain_states[-1]
    capsule["brain_stats"] = chatbot.replier.brain_stats[-1]
    capsule["reply"] = say
    chatbot.capsules_submitted.append(brain_response_to_json(capsule))
    chatbot.say_history.append(reply)
    chatbot.capsules_suggested.append(capsule_user)

    return form, reply, capsule, capsule_user


def begin_form(form, chatbot):
    capsule = {
        "chat": chatbot.chat_id,
        "turn": chatbot.turns,
        "author": chatbot.speaker,
        "utterance": "",
        "utterance_type": "STATEMENT",
        "position": "",
        "subject": {
            "label": chatbot.speaker,
            "type": ["person"],
            "uri": f"http://cltl.nl/leolani/world/{chatbot.speaker}"
        },
        "predicate": {
            "label": "know",
            "uri": f"http://cltl.nl/leolani/n2mu/know"
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
        },
        "context_id": context_id,
        "date": datetime.now().date().isoformat(),
        "place": "office",
        "place_id": place_id,
        "country": location['country'],
        "region": location['region'],
        "city": location['city'],
        "objects": [],
        "people": []
    }

    # use capsule user to fill in next form
    form = capsule_to_form(capsule, form)
    reply = {'say': chatbot.greet}

    # arrange all response info to be saved
    chatbot.capsules_suggested.append(capsule)

    return form, reply, capsule
