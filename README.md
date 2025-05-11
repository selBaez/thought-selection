# thought-selection

Repository for experiments on thought selection

## Overview

In this directory we provide the code and data for a capsule based chatbot that uses Reinforcement Learning to select
thoughts in a conversation.

The code base consists of the following files:

| Files              | Description                                                                          |
|--------------------|:-------------------------------------------------------------------------------------|
| run_experiments.py | Trains chatbot with user model conversations for dialogue policy learning.           |
| run_evaluation.py  | Evaluates the dialogue policies of trained chatbots by conversing with a user model. |
| web_interaction.py | Allows for real user interaction with the chatbot via a web app.                     |
| requirements.txt   | Requirements file.                                                                   |

<p> The implementations of the chatbot functionality and the UI for ease of interaction are divided in different folders:</p>

| Folders            | Description                                                                            |
|--------------------|:---------------------------------------------------------------------------------------|
| \\dialogue_system  | Implements a triple-based chatbot. It contains a Replier and a Response Selector.      |
| \\user_interface   | Implements a backend API and web front end to interact with the chatbot.               |
| \\user_model       | Creates different user models, as databases, with different knowledge characteristics. |
| \\results_analysis | Gathers experiment results and calculates relevant statistics and plots.               |

## Getting started

In order to run the code, follow these steps:

1) Create a virtual environment for the project (conda, venv, etc)

```bash
conda create --name thought-selection python=3.8
conda activate thought-selection
```

1) Install the required dependencies in `requirements.txt` using a regular `pip install -r requirements.txt`

```bash
pip install -r requirements.txt --no-cache
```

1) Install the latest versions of the required cltl packages. We are
   using [cltl.brain](https://github.com/leolani/cltl-knowledgerepresentation),
   [cltl.reply-generation](https://github.com/leolani/cltl-languagegeneration)
   and [cltl.dialogue-evaluation](https://github.com/leolani/cltl-dialogueevaluation). Please clone the repositories,
   pull the latest versions and install the packages into the virtual env like this:

```bash
conda activate thought-selection
cd cltl-knowledgerepresentation
git pull
pip install -e .
python -c "import nltk; nltk.download('wordnet')"
```

```bash
conda activate thought-selection
cd cltl-languagegeneration
git pull
pip install -e .
```

```bash
conda activate thought-selection
cd cltl-dialogueevaluation
git pull
pip install -e .
```

[comment]: <> (**Important:** In order to run NSP, make sure to download the NSP model and place the resource files into a)

[comment]: <> (directory `\next_sentence_prediction\model`. The model files can be found in the)

[comment]: <> (following [Google Drive folder]&#40;https://drive.google.com/drive/folders/10GEpnjqXn4DfyKjFjJG7KbJEygvdAI2J?usp=sharing&#41;.)

[comment]: <> (The code has been tested on both Windows 10 and Ubuntu 20.04.)

## Usage

Please remember to have [GraphDB Free](http://graphdb.ontotext.com/) installed and running. There should be a repository
called `sandbox`. You may also upload the `db-config.ttl` file to the triple store in order to create a repository with
the correct settings.

[comment]: <> (### Command line)

[comment]: <> (Run any of the following to begin chatting on the command line.)

[comment]: <> (**Windows:**<br>)

[comment]: <> (RL:      `$ py -3 main.py --speaker=john --mode=RL --savefile=/../../resources/thoughts.json `<br>)

[comment]: <> (NSP:    `$ py -3 main.py --speaker=john --mode=NSP --savefile=/../../resources/model `<br>)

[comment]: <> (Lenka: `$ py -3 main.py --speaker=john --mode=Lenka `)

[comment]: <> (**Ubuntu:**<br>)

[comment]: <> (RL:      `$ python3 main.py --speaker=john --mode=RL --savefile=/../../resources/thoughts.json `<br>)

[comment]: <> (NSP:    `$ python3 main.py --speaker=john --mode=NSP --savefile=/../../resources/model `<br>)

[comment]: <> (Lenka: `$ python3 main.py --speaker=john --mode=Lenka `)

[comment]: <> (### Jupyter notebook)

[comment]: <> (Initialize a Jupyter Lab session like so:)

[comment]: <> (```bash)

[comment]: <> (cd src)

[comment]: <> (jupyter-lab)

[comment]: <> (```)

[comment]: <> (Now run the `interactive_chatbot.ipynb` notebook to begin chatting.)

### Web based

Run the web app from the `src` folder, like so:

```bash
conda activate thought-selection
cd thought-selection
cd src/
python web_interaction.py
```

Now you can access [`http://127.0.0.1:5000/`](http://127.0.0.1:5000/) to begin chatting. The client may be blocked in
newer versions of MAC OS for Safari. In that case try Google Chrome.

## Authors

* [Selene Báez Santamaría](https://selbaez.github.io/)
* [Thomas Bellucci](https://github.com/thomas097)
* [Piek Vossen](https://github.com/piekvossen)



