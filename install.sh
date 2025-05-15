wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
chmod +x Anaconda3-2024.02-1-Linux-x86_64.sh
./Anaconda3-2024.02-1-Linux-x86_64.sh

### exit the server and go back in

conda create --name thought-selection python=3.8.18
conda activate thought-selection

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

git clone https://github.com/selBaez/thought-selection.git
cd thought-selection
pip install -r requirements.txt
cd ..

curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama run llama3.2

## on your local
scp Downloads/graphdb-desktop_10.6.3-1_amd64.deb sbaezsanta@145.38.189.135:/home/sbaezsanta/data/sbsgraph/
##

apt install default-jdk
apt-get install unzip
unzip graphdb-10.6.3-dist.zip
cd graphdb-10.6.3/bin/
bash graphdb
./console

### inside console
create graphdb
sandbox
owl2-rl-optimized #rule set
false             # disable same as
true              #enable context id
##

bash graphdb -d -s -Xmx100g

### move user models to raw
scp Documents/PhD/research/thought-selection/resources/users/raw_large sbaezsanta@145.38.189.135:/home/sbaezsanta/data/sbsgraph/thought-selection/resources/users/raw/
