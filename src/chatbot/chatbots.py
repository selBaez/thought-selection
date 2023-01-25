""" Filename:     chatbots.py
    Author(s):    Thomas Bellucci, Selene Baez Santamaria
    Description:  Implementation of the Chatbot based on a Leolani backend.
                  The implementation uses the knowledge extraction modules
                  of Leolani for parsing, the Brain for storage/querying
                  of triples and modified LenkaRepliers for phrasing.                   
    Date created: Nov. 11th, 2021
"""

import json
import os
from random import choice

# Pip-installed ctl repositories
from cltl.brain.long_term_memory import LongTermMemory
from cltl.brain.utils.helper_functions import brain_response_to_json
from cltl.commons.casefolding import casefold_capsule
from cltl.commons.discrete import UtteranceType
from cltl.commons.language_data.sentences import (GOODBYE, GREETING, SORRY, TALK_TO_ME)

from src.chatbot.replier import RLCapsuleReplier
from src.chatbot.utils.global_variables import BASE_CAPSULE

# Set up Java PATH (required for Windows)
os.environ["JAVAHOME"] = "C:/Program Files/Java/jre1.8.0_311/bin/java.exe"


class Chatbot:
    def __init__(self):
        """Sets up a Chatbot with a Leolani backend.

        returns: None
        """

    @property
    def replier(self):
        """Provides access to the replier."""
        return self._replier

    @property
    def brain(self):
        """Provides access to the brain."""
        return self._brain

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

    def begin_session(self, chat_id, speaker, reward, scenario_folder):
        """Sets up a session .

        params
        str chat_id:  id of chat
        str speaker:  name of speaker
        str reward:     method used to use as reward for RL
        str scenario_folder: path to write session data.

        returns: None
        """
        # Set up Leolani backend modules
        self.address = "http://localhost:7200/repositories/sandbox"
        self._brain = LongTermMemory(address=self.address, log_dir=scenario_folder, clear_all=chat_id == 1)

        # Chat information
        self.chat_id = chat_id
        self.speaker = speaker
        self.turns = 0

        # data to be recreate conversation
        self.scenario_folder = scenario_folder
        self.capsules_file = self.scenario_folder / "capsules.json"
        self.capsules_submitted = []
        self.capsules_suggested = []
        self.say_history = []

        # RL information
        self.thoughts_file = self.scenario_folder / "thoughts.json"
        self._replier = RLCapsuleReplier(self._brain, self.thoughts_file, reward)

        if chat_id == 1:
            if self.capsules_file.exists():
                self.capsules_file.unlink()
            if self.thoughts_file.exists():
                self.thoughts_file.unlink()

    def close_session(self):
        """Ends interaction and writes all learnt thought utility files.

        returns: None
        """
        # Writes a utilities JSON to disk
        self.replier.thought_selector.save(self.thoughts_file)

        # Write capsules file
        with open(self.capsules_file, "w") as file:
            json.dump(self.capsules_submitted, file, indent=4)

        # Plot
        self.replier.thought_selector.plot(filename=self.scenario_folder)

    def situate_chat(self, capsule_for_context):
        self._brain.capsule_context(capsule_for_context)

    def respond(self, capsule, return_br=True):
        """Parses the user input (as a capsule), queries and/or updates the brain
        and returns a reply by consulting the replier.

        params
        str capsule:     input capsule of the user, e.g. a response to a Thought
        bool return_br: whether to return to brain response alongside the reply

        returns: reply to input
        """
        self.turns += 2

        # ERROR
        say, capsule_user, brain_response = None, None, None
        if capsule is None:
            say = choice(SORRY) + " I could not parse that. Can you rephrase?"

        # QUESTION
        elif capsule["utterance_type"] in ["QUESTION", UtteranceType.QUESTION]:
            # Query Brain -> try to answer
            brain_response = self._brain.query_brain(casefold_capsule(capsule))
            brain_response = brain_response_to_json(brain_response)
            self._replier.reward_thought()
            say, capsule_user = self._replier.reply_to_question(brain_response), BASE_CAPSULE

        # STATEMENT
        elif capsule["utterance_type"] in ["STATEMENT", UtteranceType.STATEMENT]:
            # Update Brain -> communicate a thought
            brain_response = self._brain.capsule_statement(capsule, reason_types=True, create_label=True)
            brain_response = brain_response_to_json(brain_response)
            self._replier.reward_thought()
            say, capsule_user = self._replier.reply_to_statement(brain_response, persist=True)

        if return_br:
            return say, capsule_user, brain_response
        return say, capsule_user
