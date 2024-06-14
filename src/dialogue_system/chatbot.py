""" Filename:     chatbot.py
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
from dialogue_system.d2q_selector import D2Q
from dialogue_system.triple_phraser import TriplePhraser
from dialogue_system.utils.capsule_utils import template_to_statement_capsule
from dialogue_system.utils.global_variables import BASE_CAPSULE, BRAIN_ADDRESS, ONTOLOGY_DETAILS
from dialogue_system.utils.helpers import create_session_folder
from dialogue_system.utils.rl_parameters import RESET_FREQUENCY

# Set up Java PATH (required for Windows)
os.environ["JAVAHOME"] = "C:/Program Files/Java/jre1.8.0_311/bin/java.exe"


class Chatbot(object):
    def __init__(self):
        """Sets up a Chatbot with a Leolani backend.

        returns: None
        """
        self.scenario_folder = None
        self._brain = None
        self.chat_id = None
        self.speaker = None
        self.turns = None
        self.plots_folder = None
        self.chat_history = None
        self.thoughts_file = None
        self._selector = None
        self.statistics_history = None
        self._replier = None

    @property
    def selector(self):
        """Provides access to the selector."""
        return self._selector

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

    def begin_session(self, experiment_id, run_id, context_id, chat_id, speaker, reward, init_brain, test_model=None):
        """Sets up a session .

        params
        str chat_id:  id of chat
        str speaker:  name of speaker
        str reward:     method used to use as reward for RL

        returns: None
        """
        # Set up Leolani backend modules
        self.scenario_folder = create_session_folder(experiment_id, run_id, context_id, reward, chat_id, speaker)

        # Initialize brain
        shuffling_brain = init_brain != "None"
        if shuffling_brain:
            self._brain = LongTermMemory(address=BRAIN_ADDRESS, log_dir=self.scenario_folder,
                                         ontology_details=ONTOLOGY_DETAILS, clear_all=True)
            f = open(init_brain, "r")
            data = f.read()
            code = self.brain._upload_to_brain(data)
        else:
            resetting_brain = (chat_id - 1) % RESET_FREQUENCY == 0
            self._brain = LongTermMemory(address=BRAIN_ADDRESS, log_dir=self.scenario_folder,
                                         ontology_details=ONTOLOGY_DETAILS, clear_all=resetting_brain)

        self.brain.thought_generator._ONE_TO_ONE_PREDICATES = ['gender', 'lineage']

        # Chat information
        self.chat_id = chat_id
        self.speaker = speaker
        self.turns = 0

        # data to recreate conversation
        self.chat_history = {"capsules_submitted": [], "capsules_suggested": [], "say_history": []}
        self.plots_folder = self.scenario_folder / "plots/"
        self.plots_folder.mkdir(parents=True, exist_ok=True)

        # RL information
        prev_chat_model = None
        if test_model:
            prev_chat_model = test_model
        elif chat_id != 1:
            # load saved model from previous chat
            prev_chat_model = create_session_folder(experiment_id, run_id, context_id - 100, reward, chat_id - 1,
                                                    speaker)
            prev_chat_model = f"{prev_chat_model}/thoughts.pt"

        self._selector = D2Q(self._brain, reward=reward, trained_model=prev_chat_model,
                             states_folder=self.scenario_folder / "cumulative_states/")
        self.thoughts_file = self.scenario_folder / "thoughts.pt"
        self.statistics_history = []
        self._replier = TriplePhraser()

        if chat_id == 1:
            if self.thoughts_file.exists():
                self.thoughts_file.unlink()

    def close_session(self):
        """Ends interaction and writes all learnt thought utility files.

        returns: None
        """
        # Writes utilities JSON to disk
        self._selector.save(self.thoughts_file)

        # Save chat history
        history = self.report_chat_to_json()

        # Plot
        # self._selector.plot(history, plots_folder=self.scenario_folder / f"plots/")

    def report_chat_to_json(self):
        # Write capsules file
        with open(self.scenario_folder / "chat_history.json", "w") as file:
            json.dump(self.chat_history, file, indent=4)

        # Write brain statistics file
        with open(self.scenario_folder / "state_history.json", "w") as file:
            json.dump(self.statistics_history, file, indent=4)

        # Write chat history file
        with open(self.scenario_folder / "selection_history.json", "w") as file:
            # cast for json
            action_history = []
            for el in self.selector.action_history:
                if el:
                    action_history.append(int(el[0][0]))
                else:
                    action_history.append(None)

            hist = {"actions": action_history,
                    "rewards": self.selector.reward_history,
                    "states": self.selector.state_history["metrics"],
                    "selections": self.selector.selection_history}
            json.dump(hist, file, indent=4)

        return hist

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
        say, response_template, brain_response = None, None, None
        if capsule is None:
            say = choice(SORRY) + " I could not parse that. Can you rephrase?"

        # QUESTION
        elif capsule["utterance_type"] in ["QUESTION", UtteranceType.QUESTION]:
            # Query Brain -> try to answer
            brain_response = self._brain.query_brain(casefold_capsule(capsule))
            brain_response = brain_response_to_json(brain_response)
            self._selector.reward_thought()
            brain_response["thoughts"] = self._selector.select(brain_response)
            say, response_template = self._replier.reply_to_question(brain_response), BASE_CAPSULE

        # STATEMENT
        elif capsule["utterance_type"] in ["STATEMENT", UtteranceType.STATEMENT]:
            # Update Brain -> communicate a thought
            brain_response = self._brain.capsule_statement(capsule, reason_types=True, create_label=True)
            brain_response = brain_response_to_json(brain_response)
            # Calculate brain state
            self._selector.reward_thought()
            profile = self.selector.state_evaluator.calculate_brain_statistics(brain_response)
            self.statistics_history.append(profile)

            brain_response["thoughts"] = self._selector.select(brain_response)
            say, response_template = self._replier.reply_to_statement(brain_response, persist=True)
            response_template = template_to_statement_capsule(response_template, self)

        # Add information to capsule
        capsule["last_reward"] = self.selector.reward_history[-1]
        capsule["brain_state"] = self.selector.state_history["metrics"][-1]
        capsule["statistics_history"] = self.statistics_history[-1]
        capsule["reply"] = say

        # Keep track of everything
        self.chat_history["capsules_submitted"].append(brain_response_to_json(capsule))
        self.chat_history["say_history"].append(say)
        self.chat_history["capsules_suggested"].append(response_template)

        if return_br:
            return say, response_template, brain_response
        return say, response_template
