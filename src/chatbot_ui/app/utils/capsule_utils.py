from datetime import datetime
from random import getrandbits

import requests

context_id = getrandbits(8)
place_id = getrandbits(8)
location = requests.get("https://ipinfo.io").json()


def capsule_to_form(capsule, form):
    form.subject_label.data = capsule["subject"]["label"]
    form.subject_types.data = ','.join(capsule["subject"]["type"])
    form.predicate_label.data = capsule["predicate"]["label"]
    form.object_label.data = capsule["object"]["label"]
    form.object_types.data = ','.join(capsule["object"]["type"])
    form.perspective_certainty.data = capsule["perspective"]["certainty"]
    form.perspective_polarity.data = capsule["perspective"]["polarity"]
    form.perspective_sentiment.data = capsule["perspective"]["sentiment"]

    form.context_id.data = capsule["context_id"]
    # form.context_date.data = capsule["date"]# datetime.strftime(capsule["date"], "%Y-%m-%d")
    form.place_label.data = capsule["place"]
    form.place_id.data = capsule["place_id"]
    form.country.data = capsule["country"]
    form.region.data = capsule["region"]
    form.city.data = capsule["city"]
    # form.objects.data = ','.join(capsule["objects"])
    # form.people.data = '.'.join(capsule["people"])

    return form


def form_to_capsule(form, chatbot):
    capsule = {}
    capsule["chat"] = chatbot._chat_id
    capsule["turn"] = chatbot._turns
    capsule["author"] = chatbot._speaker
    capsule["utterance"] = ""
    capsule["utterance_type"] = "STATEMENT"
    capsule["position"] = ""

    capsule["subject"] = {"label": form.subject_label.data,
                          "type": form.subject_types.data.split(','),
                          "uri": f"http://cltl.nl/leolani/world/{form.subject_label.data}"}
    capsule["predicate"] = {"label": form.predicate_label.data,
                            "uri": f"http://cltl.nl/leolani/n2mu/{form.predicate_label.data}"}
    capsule["object"] = {"label": form.object_label.data,
                         "type": form.object_types.data.split(','),
                         "uri": f"http://cltl.nl/leolani/world/{form.object_label.data}"}
    capsule["perspective"] = {"certainty": form.perspective_certainty.data,
                              "polarity": form.perspective_polarity.data,
                              "sentiment": form.perspective_sentiment.data}

    capsule["context_id"] = form.context_id.data
    capsule["date"] = "2021-03-12" # datetime.strptime(form.context_date.data, "%Y-%m-%d")
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
    capsule = form_to_capsule(form, chatbot)

    # process with brain
    say, capsule_user, brain_response = chatbot.respond(capsule)

    # use capsule user to fill in next form
    form = capsule_to_form(capsule_user, form)

    reply = {'say': say}

    return form, reply, capsule, capsule_user


def begin_form(form, chatbot):
    capsule = {
        "chat": chatbot._chat_id,
        "turn": chatbot._turns,
        "author": chatbot._speaker,
        "utterance": "",
        "utterance_type": "STATEMENT",
        "position": "",
        "subject": {
            "label": chatbot._speaker,
            "type": ["person"],
            "uri": f"http://cltl.nl/leolani/world/{chatbot._speaker}"
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
        "date": "2021-03-12",#datetime.now().date(),
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

    return form, reply, capsule
