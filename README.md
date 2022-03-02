# thought-selection

Repository for experiments on thought selection in the @Leolani framework

## Overview

In this directory we provide the code and data for a capsule based chatbot that uses Reinforcement Learning to select
thoughts in a conversation.

The code base consists of the following files:

| Files                   | Description   |
| ----------------------- |:--------------|
| main.py                 | Runs an interaction with the chatbot via command line.|
| interactive_chatbot.py  | Runs an interaction with the chatbot via a Jupyter notebook|
| web_app.py              | Runs an interaction with the chatbot via a web app|
| requirements.txt        | Requirements file containing the minimum number of packages needed to run the implementation. |

<p> The implementations of the chatbot functionality and the UI for ease of interaction are divided in different folders:</p>

| Folders                   | Description     |
| ------------------------- | :-------------- |
| \\chatbot                 | Implements a capsule chatbot based on the Leolani brain. It contains a Thought Selector|
| \\chatbot_ui              | Implements a backend API and web front end to interact with the capsule chatbot in a more user-friendly way . |

## Getting started

In order to run the code, install the required dependencies in `requirements.txt`
using `pip install -r requirements.txt`; then run one of the following commands in the `src` directory:

```bash
conda create --name thought-selection python=3.7
conda activate thought-selection
pip install --upgrade pip
pip install -r requirements.txt --no-cache
python -m ipykernel install --name=thought-selection
```

**Important** We are using the latest versions of [cltl.brain](https://github.com/leolani/cltl-knowledgerepresentation)
and [cltl.reply-generation](https://github.com/leolani/cltl-languagegeneration) so please clone the repo, pull the
latests verions and install the packages into the virtual env.

**Important:** In order to run NSP, make sure to download the NSP model and place the resource files into a
directory `\next_sentence_prediction\model`. The model files can be found in the
following [Google Drive folder](https://drive.google.com/drive/folders/10GEpnjqXn4DfyKjFjJG7KbJEygvdAI2J?usp=sharing).

The code has been tested on both Windows 10 and Ubuntu 20.04.

## Usage

### Command line

Run any of the following to begin chatting on the command line.

**Windows:**<br>

RL:      `$ py -3 main.py --speaker=john --mode=RL --savefile=/../../resources/thoughts.json `<br>
NSP:    `$ py -3 main.py --speaker=john --mode=NSP --savefile=/../../resources/model `<br>
Lenka: `$ py -3 main.py --speaker=john --mode=Lenka `

**Ubuntu:**<br>

RL:      `$ python3 main.py --speaker=john --mode=RL --savefile=/../../resources/thoughts.json `<br>
NSP:    `$ python3 main.py --speaker=john --mode=NSP --savefile=/../../resources/model `<br>
Lenka: `$ python3 main.py --speaker=john --mode=Lenka `

### Jupyter notebook

Initialize a Jupyter Lab session like so:

```bash
cd src
jupyter-lab
```

Now run the `interactive_chatbot.ipynb` notebook to begin chatting.

### Web based

Run the web app like so:

```bash
cd src/
python web_app.py
```

Now you can access [`http://127.0.0.1:5000/`](http://127.0.0.1:5000/) to begin chatting.

## Authors

* [Selene Báez Santamaría](https://selbaez.github.io/)
* [Thomas Bellucci](https://github.com/thomas097)
* [Piek Vossen](https://github.com/piekvossen)
