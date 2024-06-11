##################################### Set files for helper libraries #####################################
for path in /data/sbs-graphs/cltl-knowledgerepresentation/src /data/sbs-graphs/cltl-knowledgereasoning/src /data/sbs-graphs/cltl-combot/src /data/sbs-graphs/cltl-languagegeneration/src /data/sbs-graphs/thought-selection/src /data/sbs-graphs/thought-selection; do
  case :$PYTHONPATH: in
  *:"$path":*) continue ;;
  esac
  PYTHONPATH=$path${PYTHONPATH+":$PYTHONPATH"}
done

echo $PYTHONPATH

##################################### Run graphdb #####################################
# daemon (in background) and server-only (no UI)
cd /data/sbs-graphs/graphdb-10.6.3/bin/
bash graphdb -d -s -Xmx55g
cd /data/sbs-graphs/thought-selection/src/

##################################### set hyper parameters #####################################
# 25  turns, 3  chats, 8 metrics, 3 runs = 2   hours
# 100 turns, 50 chats, 1 metric , 1 run  = 6.5 hours. TOO LONG!
declare -i turns=25
declare -i chats=6
declare -i runs=1

#################################### Sparseness #####################################
reward="Sparseness"
declare -i setting_id=11

for run in $(seq 1 $runs); do
  r=$((run * 1000))

  for chat in $(seq 1 $chats); do
    c=$((chat * 100))
    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"

    context_id=$((r + c + setting_id))
    python -u simulated_interaction.py \
      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward "${reward}" \
      --user_model "./../resources/users/raw/vanilla.trig" \
      --speaker "vanilla" \
      --context_id $context_id --place_id 44 --place_label "bookstore" \
      --turn_limit $turns
  done

done

###################################### Average degree #####################################
#reward="Average degree"
#declare -i setting_id=12
#
#for run in $(seq 1 $runs); do
#  r=$((run * 1000))
#
#  for chat in $(seq 1 $chats); do
#    c=$((chat * 100))
#    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"
#
#    context_id=$((r + c + setting_id))
#    python -u simulated_interaction.py \
#      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward "${reward}" \
#      --user_model "./../resources/users/raw/vanilla.trig" \
#      --speaker "vanilla" \
#      --context_id $context_id --place_id 44 --place_label "bookstore" \
#      --turn_limit $turns
#  done
#
#done
#
###################################### Shortest path #####################################
#reward="Shortest path"
#declare -i setting_id=13
#
#for run in $(seq 1 $runs); do
#  r=$((run * 1000))
#
#  for chat in $(seq 1 $chats); do
#    c=$((chat * 100))
#    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"
#
#    context_id=$((r + c + setting_id))
#    python -u simulated_interaction.py \
#      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward "${reward}" \
#      --user_model "./../resources/users/raw/vanilla.trig" \
#      --speaker "vanilla" \
#      --context_id $context_id --place_id 44 --place_label "bookstore" \
#      --turn_limit $turns
#  done
#
#done
#
###################################### Total triples #####################################
#reward="Total triples"
#declare -i setting_id=14
#
#for run in $(seq 1 $runs); do
#  r=$((run * 1000))
#
#  for chat in $(seq 1 $chats); do
#    c=$((chat * 100))
#    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"
#
#    context_id=$((r + c + setting_id))
#    python -u simulated_interaction.py \
#      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward "${reward}" \
#      --user_model "./../resources/users/raw/vanilla.trig" \
#      --speaker "vanilla" \
#      --context_id $context_id --place_id 44 --place_label "bookstore" \
#      --turn_limit $turns
#  done
#
#done
#
###################################### Average population #####################################
#reward="Average population"
#declare -i setting_id=21
#
#for run in $(seq 1 $runs); do
#  r=$((run * 1000))
#
#  for chat in $(seq 1 $chats); do
#    c=$((chat * 100))
#    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"
#
#    context_id=$((r + c + setting_id))
#    python -u simulated_interaction.py \
#      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward "${reward}" \
#      --user_model "./../resources/users/raw/vanilla.trig" \
#      --speaker "vanilla" \
#      --context_id $context_id --place_id 44 --place_label "bookstore" \
#      --turn_limit $turns
#  done
#
#done
#
###################################### Ratio claims to triples #####################################
#reward="Ratio claims to triples"
#declare -i setting_id=31
#
#for run in $(seq 1 $runs); do
#  r=$((run * 1000))
#
#  for chat in $(seq 1 $chats); do
#    c=$((chat * 100))
#    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"
#
#    context_id=$((r + c + setting_id))
#    python -u simulated_interaction.py \
#      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward "${reward}" \
#      --user_model "./../resources/users/raw/vanilla.trig" \
#      --speaker "vanilla" \
#      --context_id $context_id --place_id 44 --place_label "bookstore" \
#      --turn_limit $turns
#  done
#
#done
#
###################################### Ratio perspectives to claims #####################################
#reward="Ratio perspectives to claims"
#declare -i setting_id=32
#
#for run in $(seq 1 $runs); do
#  r=$((run * 1000))
#
#  for chat in $(seq 1 $chats); do
#    c=$((chat * 100))
#    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"
#
#    context_id=$((r + c + setting_id))
#    python -u simulated_interaction.py \
#      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward "${reward}" \
#      --user_model "./../resources/users/raw/vanilla.trig" \
#      --speaker "vanilla" \
#      --context_id $context_id --place_id 44 --place_label "bookstore" \
#      --turn_limit $turns
#  done
#
#done
#
###################################### Ratio conflicts to claims #####################################
#reward="Ratio conflicts to claims"
#declare -i setting_id=33
#
#for run in $(seq 1 $runs); do
#  r=$((run * 1000))
#
#  for chat in $(seq 1 $chats); do
#    c=$((chat * 100))
#    echo "REWARD: ${reward}, RUN: ${run}, CHAT: ${chat}"
#
#    context_id=$((r + c + setting_id))
#    python -u simulated_interaction.py \
#      --experiment_id "e1" --run_id "run${run}" --chat_id $chat --reward "${reward}" \
#      --user_model "./../resources/users/raw/vanilla.trig" \
#      --speaker "vanilla" \
#      --context_id $context_id --place_id 44 --place_label "bookstore" \
#      --turn_limit $turns
#  done
#
#done

##################################### Stop graphdb #####################################
# if running and needs to be killed
kill $(lsof -i:7200)
