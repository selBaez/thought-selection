export PYTHONPATH="${PYTHONPATH}:/Users/sbaez/Documents/PhD/research/thought-selection"
export PYTHONPATH="${PYTHONPATH}:/Users/sbaez/Documents/PhD/leolani/cltl-knowledgerepresentation/src"
export PYTHONPATH="${PYTHONPATH}:/Users/sbaez/Documents/PhD/leolani/cltl-knowledgereasoning/src"
export PYTHONPATH="${PYTHONPATH}:/Users/sbaez/Documents/PhD/leolani/cltl-combot/src"
export PYTHONPATH="${PYTHONPATH}:/Users/sbaez/Documents/PhD/leolani/cltl-languagegeneration/src"

conda activate thought-selection

# run graphdb as daemon (in background) and server-only (no UI)
#cd ~/data/sbs-graphs/graphdb-10.6.3/bin/
#bash graphdb -d -s


# TODO change to real numbers: 100 turns 500 chats 10 runs
declare -i turns=10
declare -i chats=5
declare -i runs=3


# loop to have several runs per setting (first one reward - all chats - all runs, then other rewards). Change chat and context id
declare -i setting_id=111
for run in $(seq 1 $runs); do
  run=$((r * 1000))
  echo $run

  for chat in $(seq 1 $chats); do
    context_id=$((run + setting_id))
    echo $chat
    python simulated_interaction.py \
      --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
      --speaker "John" \
      --chat_id $chat --reward "Sparseness" --turn_limit $turns \
      --context_id $context_id --place_id 44 --place_label "bookstore"
  done

done

#python simulated_interaction.py \
#  --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
#  --speaker "John" \
#  --chat_id 1 --reward "Average degree" --turn_limit $turns \
#  --context_id 121 --place_id 44 --place_label "bookstore"
#
#python simulated_interaction.py \
#  --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
#  --speaker "John" \
#  --chat_id 1 --reward "Shortest path" --turn_limit $turns \
#  --context_id 131 --place_id 44 --place_label "bookstore"
#
#python simulated_interaction.py \
#  --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
#  --speaker "John" \
#  --chat_id 1 --reward "Total triples" --turn_limit $turns \
#  --context_id 141 --place_id 44 --place_label "bookstore"
#
#python simulated_interaction.py \
#  --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
#  --speaker "John" \
#  --chat_id 1 --reward "Average population" --turn_limit $turns \
#  --context_id 211 --place_id 44 --place_label "bookstore"
#
#python simulated_interaction.py \
#  --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
#  --speaker "John" \
#  --chat_id 1 --reward "Ratio claims to triples" --turn_limit $turns \
#  --context_id 311 --place_id 44 --place_label "bookstore"
#
#python simulated_interaction.py \
#  --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
#  --speaker "John" \
#  --chat_id 1 --reward "Ratio perspectives to claims" --turn_limit $turns \
#  --context_id 321 --place_id 44 --place_label "bookstore"
#
#python simulated_interaction.py \
#  --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
#  --speaker "John" \
#  --chat_id 1 --reward "Ratio conflicts to claims" --turn_limit $turns \
#  --context_id 331 --place_id 44 --place_label "bookstore"


# if running and needs to be killed
#kill $(lsof -i:7200)