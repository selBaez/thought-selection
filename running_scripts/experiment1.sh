##################################### Set files for helper libraries #####################################
export PYTHONPATH="${PYTHONPATH}:/Users/sbaez/Documents/PhD/research/thought-selection"
export PYTHONPATH="${PYTHONPATH}:/Users/sbaez/Documents/PhD/leolani/cltl-knowledgerepresentation/src"
export PYTHONPATH="${PYTHONPATH}:/Users/sbaez/Documents/PhD/leolani/cltl-knowledgereasoning/src"
export PYTHONPATH="${PYTHONPATH}:/Users/sbaez/Documents/PhD/leolani/cltl-combot/src"
export PYTHONPATH="${PYTHONPATH}:/Users/sbaez/Documents/PhD/leolani/cltl-languagegeneration/src"

echo $PYTHONPATH

##################################### Activate virtual env #####################################
conda activate thought-selection
cd ../src/

##################################### set hyper parameters #####################################
# TODO change to real numbers: 100 turns 500 chats 10 runs
declare -i turns=5
declare -i chats=2
declare -i runs=2

##################################### Sparseness #####################################
reward="Sparseness"
declare -i setting_id=11

for run in $(seq 1 $runs); do
  r=$((run * 1000))

  for chat in $(seq 1 $chats); do
    c=$((chat * 100))
    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"

    context_id=$((r + c + setting_id))
    python simulated_interaction.py \
      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward $reward \
      --user_model "./../resources/users/raw/vanilla.trig" \
      --speaker "vanilla" \
      --context_id $context_id --place_id 44 --place_label "bookstore" \
      --turn_limit $turns
  done

done

##################################### Average degree #####################################
reward="Average degree"
declare -i setting_id=12

for run in $(seq 1 $runs); do
  r=$((run * 1000))

  for chat in $(seq 1 $chats); do
    c=$((chat * 100))
    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"

    context_id=$((r + c + setting_id))
    python simulated_interaction.py \
      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward $reward \
      --user_model "./../resources/users/raw/vanilla.trig" \
      --speaker "vanilla" \
      --context_id $context_id --place_id 44 --place_label "bookstore" \
      --turn_limit $turns
  done

done

##################################### Shortest path #####################################
reward="Shortest path"
declare -i setting_id=13

for run in $(seq 1 $runs); do
  r=$((run * 1000))

  for chat in $(seq 1 $chats); do
    c=$((chat * 100))
    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"

    context_id=$((r + c + setting_id))
    python simulated_interaction.py \
      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward $reward \
      --user_model "./../resources/users/raw/vanilla.trig" \
      --speaker "vanilla" \
      --context_id $context_id --place_id 44 --place_label "bookstore" \
      --turn_limit $turns
  done

done

##################################### Total triples #####################################
reward="Total triples"
declare -i setting_id=14

for run in $(seq 1 $runs); do
  r=$((run * 1000))

  for chat in $(seq 1 $chats); do
    c=$((chat * 100))
    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"

    context_id=$((r + c + setting_id))
    python simulated_interaction.py \
      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward $reward \
      --user_model "./../resources/users/raw/vanilla.trig" \
      --speaker "vanilla" \
      --context_id $context_id --place_id 44 --place_label "bookstore" \
      --turn_limit $turns
  done

done

##################################### Average population #####################################
reward="Average population"
declare -i setting_id=21

for run in $(seq 1 $runs); do
  r=$((run * 1000))

  for chat in $(seq 1 $chats); do
    c=$((chat * 100))
    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"

    context_id=$((r + c + setting_id))
    python simulated_interaction.py \
      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward $reward \
      --user_model "./../resources/users/raw/vanilla.trig" \
      --speaker "vanilla" \
      --context_id $context_id --place_id 44 --place_label "bookstore" \
      --turn_limit $turns
  done

done

##################################### Ratio claims to triples #####################################
reward="Ratio claims to triples"
declare -i setting_id=31

for run in $(seq 1 $runs); do
  r=$((run * 1000))

  for chat in $(seq 1 $chats); do
    c=$((chat * 100))
    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"

    context_id=$((r + c + setting_id))
    python simulated_interaction.py \
      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward $reward \
      --user_model "./../resources/users/raw/vanilla.trig" \
      --speaker "vanilla" \
      --context_id $context_id --place_id 44 --place_label "bookstore" \
      --turn_limit $turns
  done

done

##################################### Ratio perspectives to claims #####################################
reward="Ratio perspectives to claims"
declare -i setting_id=32

for run in $(seq 1 $runs); do
  r=$((run * 1000))

  for chat in $(seq 1 $chats); do
    c=$((chat * 100))
    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"

    context_id=$((r + c + setting_id))
    python simulated_interaction.py \
      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward $reward \
      --user_model "./../resources/users/raw/vanilla.trig" \
      --speaker "vanilla" \
      --context_id $context_id --place_id 44 --place_label "bookstore" \
      --turn_limit $turns
  done

done

##################################### Ratio conflicts to claims #####################################
reward="Ratio conflicts to claims"
declare -i setting_id=33

for run in $(seq 1 $runs); do
  r=$((run * 1000))

  for chat in $(seq 1 $chats); do
    c=$((chat * 100))
    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"

    context_id=$((r + c + setting_id))
    python simulated_interaction.py \
      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward $reward \
      --user_model "./../resources/users/raw/vanilla.trig" \
      --speaker "vanilla" \
      --context_id $context_id --place_id 44 --place_label "bookstore" \
      --turn_limit $turns
  done

done
