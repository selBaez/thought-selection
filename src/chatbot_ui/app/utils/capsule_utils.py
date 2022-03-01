def capsule_to_form(capsule, form):
    form.subject_label.data = capsule["subject"]["label"]
    form.subject_types.data = ','.join(capsule["subject"]["type"])
    form.predicate_label.data = capsule["predicate"]["label"]
    form.object_label.data = capsule["object"]["label"]
    form.object_types.data = ','.join(capsule["object"]["type"])
    form.perspective_certainty.data = capsule["perspective"]["certainty"]
    form.perspective_polarity.data = capsule["perspective"]["polarity"]
    form.perspective_sentiment.data = capsule["perspective"]["sentiment"]

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
                          "type": form.subject_types.data.split(',')}
    capsule["predicate"] = {"label": form.predicate_label.data}
    capsule["object"] = {"label": form.object_label.data,
                         "type": form.object_types.data.split(',')}
    capsule["perspective"] = {"certainty": form.perspective_certainty.data,
                              "polarity": form.perspective_polarity.data,
                              "sentiment": form.perspective_sentiment.data}

    capsule["context_id"] = 170
    capsule["date"] = "2021-03-12"
    capsule["place"] = "office"
    capsule["place_id"] = 98
    capsule["country"] = "NL"
    capsule["region"] = "North Holland"
    capsule["city"] = "Amsterdam"
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
            "label": "selene",
            "type": ["person"],
            "uri": f"http://cltl.nl/leolani/world/selene"
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
        "context_id": 170,
        "date": "2021-03-12",
        "place": "office",
        "place_id": 98,
        "country": "NL",
        "region": "North Holland",
        "city": "Amsterdam",
        "objects": [],
        "people": []
    }

    # use capsule user to fill in next form
    form = capsule_to_form(capsule, form)

    say = 'Nice to meet you, what is your name?'
    reply = {'say': say}

    return form, reply, capsule