""" Filename:     chatbots.py
    Author(s):    Thomas Bellucci, Selene Baez Santamaria
    Description:  Implementation of the Chatbot based on a Leolani backend.
                  The implementation uses the knowledge extraction modules
                  of Leolani for parsing, the Brain for storage/querying
                  of triples and modified LenkaRepliers for phrasing.                   
    Date created: Nov. 11th, 2021
"""

# Set up Java PATH (required for Windows)
import os

os.environ["JAVAHOME"] = "C:/Program Files/Java/jre1.8.0_311/bin/java.exe"

from pathlib import Path
from random import choice

# Pip-installed ctl repositories
from cltl.brain.long_term_memory import LongTermMemory
from cltl.brain.utils.helper_functions import brain_response_to_json
from cltl.combot.backend.api.discrete import UtteranceType
from cltl.reply_generation.data.sentences import (GOODBYE, GREETING, SORRY, TALK_TO_ME)

from src.chatbot.replier import RLCapsuleReplier
from src.chatbot.utils.chatbot_utils import capsule_for_query
from src.chatbot.utils.thoughts_utils import copy_capsule_context, BASE_CAPSULE


class Chatbot:
    def __init__(self, chat_id, speaker, mode, reward, savefile=None):
        """Sets up a Chatbot with a Leolani backend.

        params
        str speaker:  name of speaker
        str mode:     method used to select thoughts: ['Lenka', 'RL', 'NSP']
        str savefile: path to NSP model or utilities file needed by the replier.

        returns: None
        """
        # Set up Leolani backend modules
        self._address = "http://localhost:7200/repositories/sandbox"
        self._brain = LongTermMemory(address=self._address, log_dir=Path("./../../logs"), clear_all=False)

        self._mode = mode
        self._savefile = savefile
        self._chat_id = chat_id
        self._speaker = speaker
        self._turns = 0

        if mode == "RL":
            self._replier = RLCapsuleReplier(self._brain, Path(savefile), reward)
        else:
            raise Exception("unknown replier mode %s (choose RL)" % mode)

    @property
    def replier(self):
        """Provides access to the replier."""
        return self._replier

    @property
    def greet(self):
        """Generates a random greeting."""
        string = choice(GREETING) + " " + choice(TALK_TO_ME)
        return string

    @property
    def farewell(self):
        """Generates a random farewell message."""
        string = choice(GOODBYE)
        return string

    def close(self):
        """Ends interaction and writes all learnt thought utility files
        (if method='RL') .

        returns: None
        """
        # Writes (optionally) a utilities JSON to disk
        if self._savefile and self._mode == "RL":
            self._replier.thought_selector.save(self._savefile)

    def respond(self, capsule, return_br=True):
        """Parses the user input (as a capsule), queries and/or updates the brain
        and returns a reply by consulting the replier.

        params
        str capsule:     input capsule of the user, e.g. a response to a Thought
        bool return_br: whether to return to brain response alongside the reply

        returns: reply to input
        """
        self._turns += 1

        # ERROR
        say, capsule_user, brain_response = None, None, None
        if capsule is None:
            say = choice(SORRY) + " I could not parse that. Can you rephrase?"

        # QUESTION
        elif capsule["utterance_type"] in ["QUESTION", UtteranceType.QUESTION]:
            # Query Brain -> try to answer
            brain_response = self._brain.query_brain(capsule_for_query(capsule))
            brain_response = brain_response_to_json(brain_response)

            if isinstance(self._replier, RLCapsuleReplier):
                self._replier.reward_thought()

            say, capsule_user = self._replier.reply_to_question(brain_response), BASE_CAPSULE

        # STATEMENT
        elif capsule["utterance_type"] in ["STATEMENT", UtteranceType.STATEMENT]:
            # Update Brain -> communicate a thought
            brain_response = self._brain.update(capsule, reason_types=True, create_label=True)
            brain_response = brain_response_to_json(brain_response)

            if isinstance(self._replier, RLCapsuleReplier):
                self._replier.reward_thought()

            say, capsule_user = self._replier.reply_to_statement(brain_response)
            # say, capsule_user = self._select_rl_thought(brain_response)

        if capsule_user:
            capsule_user['chat'] = self._chat_id
            capsule_user['turn'] = self._turns + 1
            capsule_user['author'] = self._speaker
            capsule_user['utterance_type'] = "STATEMENT"
            capsule_user = copy_capsule_context(capsule_user, brain_response['statement'])

        if return_br:
            return say, capsule_user, brain_response
        return say, capsule_user
