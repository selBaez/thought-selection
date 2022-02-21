"""
NOTES
rdf available only for group Piek, jaap
"""

import json
import os

import numpy as np
import pandas as pd
from cltl.triple_extraction.api import Chat
from cltl.triple_extraction.cfg_analyzer import CFGAnalyzer
from cltl.triple_extraction.oie_analyzer import OIEAnalyzer
from cltl.triple_extraction.utils.helper_functions import utterance_to_capsules

from src.entity_linking.label_linker import LabelBasedLinker

ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = ABSOLUTE_PATH + f'/../../data'
INPUT_FOLDER = DATA_FOLDER + f'/leolani2-evaluations/'
OUTPUT_FOLDER = DATA_FOLDER + f'/combots_convos_capsules/'

CONVOS_FILES = [
    # '/g1-piek/scenario1/2021-12-07-16-26-14_truns83_imme.csv',
    # '/g1-piek/scenario1/2021-12-07-16-26-14_turns83_fina.csv',
    '/g1-piek/scenario1/2021-12-07-16_26_14_turns83_thomas.csv',
    # '/g1-piek/scenario2/2021-12-07-16-26-14_turns57_imme.csv',
    # '/g1-piek/scenario2/2021-12-07-16-26-14_turns57_fina.csv',
    '/g1-piek/scenario2/2021-12-10-09_35_57_turns57_thomas.csv',
    # '/g2-jaap/scenario1/annotation Ella scenario 1.csv',
    '/g2-jaap/scenario1/annotation Ella scenario 1b.csv',
    '/g2-jaap/scenario1/annotation Piotr scenario 1.csv',
    # '/g2-jaap/scenario2/annotation Ella scenario 2.csv',
    # '/g2-jaap/scenario2/annotation Ella scenario 2b.csv',
    '/g2-jaap/scenario2/annotation Piotr scenario 2.csv',
    # '/g3-lea/scenario1/Caya_2021-12-10-09_35_02_turns122_context300_v1.csv',
    # '/g3-lea/scenario1/Caya_2021-12-10-09_35_02_turns122_context300_v1b.csv',
    # '/g3-lea/scenario1/Mincke_2021-12-10-09_35_02.csv',
    '/g3-lea/scenario1/Mincke_2021-12-10-09_35_02b.csv',
    # '/g3-lea/scenario2/Rishvik_and_Pauline1.csv',
    '/g3-lea/scenario2/Rishvik_and_Pauline1b.csv',
    # '/g4-tae/scenario1/nihed -1 - 2021-12-10-10_26_45_turns78_context300.csv',
    # '/g4-tae/scenario1/Fajjaaz - 1 - 2021-12-10-10_26_45_turns78_context300.csv',
    '/g4-tae/scenario1/Hidde_1.csv',
    # '/g4-tae/scenario1/nicole_1 - 2021-12-10-10_26_45_turns78_context300.csv',
    '/g4-tae/scenario2/Fajjaaz - 2 - 2021-12-10-09_23_53_turns10_context300.csv',
    '/g4-tae/scenario2/Hidde_2.csv',
    # '/g4-tae/scenario2/nicole_2 - 2021-12-10-09_23_53_turns10_context300.csv',
    # '/g4-tae/scenario2/nihed -2 - 2021-12-10-09_23_53_turns10_context300.csv
]


def guess_speaker(speaker_column):
    possible = set(speaker_column.values)
    possible.discard('Leolani')
    possible.discard('Leolani2')
    possible.discard(np.nan)
    possible = list(possible)
    possible.sort()
    speaker = possible[0]

    return speaker


def save(file_num, df, convo_capsules, convo_noisy_capsules):
    df = df[['Turn', 'Speaker', 'Cue', 'Capsules', 'Capsules OIE']]
    df.to_csv(OUTPUT_FOLDER + f'{file_num}.csv', index=False)

    f = open(OUTPUT_FOLDER + f"{file_num}.json", "w")
    json.dump(convo_capsules, f)

    f = open(OUTPUT_FOLDER + f"{file_num}_noisy.json", "w")
    json.dump(convo_noisy_capsules, f)


def extract_triples(convos_files):
    # Create couple analyzers and a linker
    analyzer_1 = CFGAnalyzer()
    analyzer_2 = OIEAnalyzer()
    linker = LabelBasedLinker()

    for file_num, file in enumerate(convos_files):
        # Read CSV with pandas
        df = pd.read_csv(INPUT_FOLDER + file, header=0)

        # Create placeholders
        df["Capsules"] = np.nan
        df["Capsules OIE"] = np.nan
        convo_capsules = []
        convo_noisy_capsules = []

        # Chat with who?
        speaker = guess_speaker(df['Speaker'])
        chat = Chat(speaker)

        # Iterate through turns
        for idx, row in df.iterrows():
            if type(row['Response']) == str:
                # Get correct speaker info
                if row['Speaker'] != speaker:
                    if type(row['Speaker']) == str and 'Leolani' in row['Speaker']:
                        chat._speaker = 'Leolani'
                    else:
                        speaker = row['Speaker']
                        chat = Chat(speaker)
                else:
                    chat._speaker = speaker

                # add utterance
                chat.add_utterance(row['Response'])

                # Try main analyzer
                analyzer_1.analyze(chat.last_utterance)
                capsules = utterance_to_capsules(chat.last_utterance)
                if capsules:
                    # Use linker to add uris to mentions
                    capsules = [linker.link(cpsl) for cpsl in capsules]
                    df.loc[idx, 'Capsules'] = str(capsules)
                    convo_capsules.extend(capsules)
                    convo_noisy_capsules.extend(capsules)
                else:
                    # Backup analyzer
                    analyzer_2.analyze(chat.last_utterance)
                    capsules = utterance_to_capsules(chat.last_utterance)
                    if capsules:
                        capsules = [linker.link(cpsl) for cpsl in capsules]
                        df.loc[idx, 'Capsules OIE'] = str(capsules)
                        convo_noisy_capsules.extend(capsules)

            # if idx == 3:
            #     break

        save(file_num, df, convo_capsules, convo_noisy_capsules)


if __name__ == "__main__":
    extract_triples(CONVOS_FILES)

    print('ALL DONE')
