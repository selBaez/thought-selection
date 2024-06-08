from datetime import datetime

from dialogue_system.utils.global_variables import CONTEXT_ID


def template_to_statement_capsule(template_capsule, chatbot):
    template_capsule['chat'] = chatbot.chat_id
    template_capsule['turn'] = chatbot.turns + 1
    template_capsule["author"] = {"label": chatbot.speaker.lower(),
                                  "type": ["person"],
                                  "uri": f"http://cltl.nl/leolani/world/{chatbot.speaker.lower()}"}
    template_capsule['utterance_type'] = "STATEMENT"
    template_capsule['context_id'] = CONTEXT_ID
    template_capsule["timestamp"] = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S.%f")

    return template_capsule


