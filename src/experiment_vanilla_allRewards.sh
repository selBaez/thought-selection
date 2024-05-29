export PYTHONPATH="${PYTHONPATH}:/Users/sbaez/Documents/PhD/research/thought-selection"
export PYTHONPATH="${PYTHONPATH}:/Users/sbaez/Documents/PhD/leolani/cltl-knowledgerepresentation/src"
export PYTHONPATH="${PYTHONPATH}:/Users/sbaez/Documents/PhD/leolani/cltl-knowledgereasoning/src"
export PYTHONPATH="${PYTHONPATH}:/Users/sbaez/Documents/PhD/leolani/cltl-combot/src"
export PYTHONPATH="${PYTHONPATH}:/Users/sbaez/Documents/PhD/leolani/cltl-languagegeneration/src"

python simulated_interaction.py \
  --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
  --speaker "John" \
  --chat_id 1 --reward "Sparseness" --turn_limit 100 \
  --context_id 111 --place_id 44 --place_label "bookstore"

python simulated_interaction.py \
  --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
  --speaker "John" \
  --chat_id 1 --reward "Average degree" --turn_limit 100 \
  --context_id 121 --place_id 44 --place_label "bookstore"

python simulated_interaction.py \
  --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
  --speaker "John" \
  --chat_id 1 --reward "Shortest path" --turn_limit 100 \
  --context_id 131 --place_id 44 --place_label "bookstore"

python simulated_interaction.py \
  --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
  --speaker "John" \
  --chat_id 1 --reward "Total triples" --turn_limit 100 \
  --context_id 141 --place_id 44 --place_label "bookstore"

python simulated_interaction.py \
  --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
  --speaker "John" \
  --chat_id 1 --reward "Average population" --turn_limit 100 \
  --context_id 211 --place_id 44 --place_label "bookstore"

python simulated_interaction.py \
  --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
  --speaker "John" \
  --chat_id 1 --reward "Ratio claims to triples" --turn_limit 100 \
  --context_id 311 --place_id 44 --place_label "bookstore"

python simulated_interaction.py \
  --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
  --speaker "John" \
  --chat_id 1 --reward "Ratio perspectives to claims" --turn_limit 100 \
  --context_id 321 --place_id 44 --place_label "bookstore"

python simulated_interaction.py \
  --user_model "/Users/sbaez/Documents/PhD/research/thought-selection/resources/users/raw/vanilla.trig" \
  --speaker "John" \
  --chat_id 1 --reward "Ratio conflicts to claims" --turn_limit 100 \
  --context_id 331 --place_id 44 --place_label "bookstore"
