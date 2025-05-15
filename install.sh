### First we set up conda
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
chmod +x Anaconda3-2024.02-1-Linux-x86_64.sh
./Anaconda3-2024.02-1-Linux-x86_64.sh
# exit the server and go back in for changes to take place
conda create --name thought-selection python=3.8.18
conda activate thought-selection

### Then we install packages we have to clone from github
cd data/sbsgraph

git clone https://github.com/leolani/cltl-combot.git
cd cltl-combot
pip install -e .
cd ..

git clone https://github.com/leolani/cltl-knowledgerepresentation.git
cd cltl-knowledgerepresentation
pip install -e .
git switch 36-separate-thought-creation
cd ..

git clone https://github.com/leolani/cltl-knowledgereasoning.git
cd cltl-knowledgereasoning
pip install -e .
cd ..

git clone https://github.com/leolani/cltl-languagegeneration.git
cd cltl-languagegeneration
pip install -e .
git switch 18-separate-thought-selection
cd ..

python -c "import nltk; nltk.download('wordnet')"

### Then we clone our repo and install standard packages
git clone https://github.com/selBaez/thought-selection.git
cd thought-selection
pip install -r requirements.txt
cd ..

### Now we do specific tool installation. Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

### Java (needed for graphDB)
sudo apt install openjdk-21-jdk
sudo apt-get install unzip

### Install graphDB from zip and set license
wget https://download.ontotext.com/owlim/b1d91bea-25d4-11f0-829f-42843b1b6b38/graphdb-11.0.1-dist.zip
unzip graphdb-11.0.1-dist.zip
# on your local machine
scp -r Downloads/UZH_GRAPHDB_FREE_v11.0.license sbaezsanta@145.38.189.135:/home/sbaezsanta/data/sbsgraph/graphdb-11.0.1/conf/
#
cd /home/sbaezsanta/data/sbsgraph/graphdb-11.0.1/conf
mv UZH_GRAPHDB_FREE_v11.0.license graphdb.license

### We need to create a repository with specific parameters
cd ../bin/
./console
# inside console
create graphdb
sandbox
owl2-rl-optimized #rule set
false             # disable same as
true              # enable context id
true              # enable predicate list
#
exit

# In case you want to test the installation went well
#bash graphdb
#kill $(lsof -i:7200)

### Now we have to replace the small data from github with the big data that we have locally
cd ../../thought-selection/resources/users
rm -r raw/
# on your local machine
scp -r Documents/PhD/research/thought-selection/resources/users/raw_large/ sbaezsanta@145.38.189.135:/home/sbaezsanta/data/sbsgraph/thought-selection/resources/users/
#
mv /home/sbaezsanta/data/sbsgraph/thought-selection/resources/users/raw_large /home/sbaezsanta/data/sbsgraph/thought-selection/resources/users/raw

### Everything is set! You can now run an experiment on the server.
# Remember to adjust the `run_experiments` script parameters
cd data/sbsgraph/thought-selection/src
vim run_experiments # i -> ESC -> :wq
