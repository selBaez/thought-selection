# thought-selection
Repository for experiments on thought selection in the @Leolani framework


## Overview

In this directory we provide the *reinforcement learning* (RL) implementations of the Leolani replier.

The code base consists of the following files:

| Files               | Description   |
| ------------------- |:--------------|
| main.py             | Runs an interaction with the chatbot. By default it runs the RL implementation, but can be changed to `NSP` or `Lenka` using the `--mode` command line argument (see Usage section).|
| evaluate_brain.py   | Exploration to evaluate the brain using graph metrics. |
| requirements.txt    | Requirements file containing the minimum number of packages needed to run the implementation. |

<p> The implementations of the RL and NSP method are divided into separate folders:</p>

| Folders                   | Description     |
| ------------------------- | :-------------- |
| \\chatbot                    | Implements a capsule chatbot based on the Leolani brain. |
| \\metrics                    | Implements graph, ontology and brain metrics. |

## Usage

In order to run the code, install the required dependencies in `requirements.txt` using `pip install -r requirements.txt`; then run one of the following commands in the `src` directory:

**Windows:**<br>

RL:      `$ py -3 main.py --speaker=john --mode=RL --savefile=/../../resources/thoughts.json `<br>
NSP:    `$ py -3 main.py --speaker=john --mode=NSP --savefile=/../../resources/model `<br>
Lenka: `$ py -3 main.py --speaker=john --mode=Lenka `

**Ubuntu:**<br>

RL:      `$ python3 main.py --speaker=john --mode=RL --savefile=/../../resources/thoughts.json `<br>
NSP:    `$ python3 main.py --speaker=john --mode=NSP --savefile=/../../resources/model `<br>
Lenka: `$ python3 main.py --speaker=john --mode=Lenka `

The code has been tested on both Windows 10 and Ubuntu 20.04.

**Important:** In order to run NSP, make sure to download the NSP model and place the resource files into a directory `\next_sentence_prediction\model`. The model files can be found in the following [Google Drive folder](https://drive.google.com/drive/folders/10GEpnjqXn4DfyKjFjJG7KbJEygvdAI2J?usp=sharing).
