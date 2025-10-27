#!/bin/bash

# set -x
MODEL_DIR=${1}
NUM_GPU=${2}

# set max_token and response_token in src/model/run_model.py line 187
cd "$(dirname "$0")/utils"
domains=($(find benchmark -mindepth 1 -maxdepth 1 -type d | xargs -n 1 basename))  # run all categories
for domain in ${domains[@]}; do
    system_tasks=($(find benchmark/${domain} -mindepth 1 -maxdepth 1 -type d | xargs -n 1 basename))  # run all tasks
    for system_task in ${system_tasks[@]}; do
        model_path=${MODEL_DIR}
        script_type="torchllms"

        folders=($(find benchmark/${domain}/${system_task} -mindepth 2 -maxdepth 2 -type d | grep -E 'aligned|conflict' | awk -F '/' '{print $(NF-1)"/"$NF}'))
        for folder in ${folders[@]}; do
            echo -e "\n\e[31m********************** ${domain}/${system_task}/${folder} **********************\e[0m"
            OUTPUT_DIR=../outputs/${OUTPUT_FOLDER}/${domain}/${system_task}/${folder}
            # For vllm only: tensor_parallel
            # For API call only: max_retries, max_threads, sleep
            python src/model/run_model.py \
                -model ${model_path} \
                -input benchmark/${domain}/${system_task}/${folder}/input_data.json \
                -request_file ${OUTPUT_DIR}/input_request.json \
                -response_file ${OUTPUT_DIR}/input_response.json \
                -eval_output_dir ${OUTPUT_DIR} \
                -max_tokens 2048 \
                -temperature 0.0 \
                -top_p 1.0 \
                -top_k 1 \
                -precision auto \
                -task ${system_task} \
                -backend ${script_type} \
                -max_retries 10 \
                -max_threads 16 \
                -tensor_parallel ${NUM_GPU} \
                -sleep 5

            echo -e "\e[32m--------------------- Process Scores ---------------------\e[0m"
            # For tool use (get-webpage task), we need to calculate the aggregated score of three NLP tasks
            # if [ "$system_task" = "get-webpage" ] ; then
            #     if [ "$folder" = "reference/default" ] ; then
            #         python src/task_execution/calc_mix_reference_score.py \
            #             -input ${OUTPUT_DIR}/eval_results.json \
            #             -record_dir ${OUTPUT_DIR}
            #     else
            #         python src/task_execution/calc_mix_task_score.py \
            #             -input ${OUTPUT_DIR}/eval_results.json

            #         python src/model/record_scores.py \
            #             -data ${OUTPUT_DIR}/eval_results.json \
            #             -output_dir ${OUTPUT_DIR}
            #     fi

            # For calculating the reference score of NLP tasks (except for language detection task)
            # elif [ "$domain" = "task-execution" ] && [ "$folder" = "reference/default" ] && [ "$system_task" != "lang-detect" ]; then
            #     python src/task_execution/calc_reference_score.py \
            #         -input ${OUTPUT_DIR}/eval_results.json \
            #         -task ${system_task} \
            #         -record_dir ${OUTPUT_DIR}
            
            # For recording the model scores to a file
            # else
            #     python src/model/record_scores.py \
            #         -data ${OUTPUT_DIR}/eval_results.json \
            #         -output_dir ${OUTPUT_DIR}
            IFS='/' read -r instruction_type prompt_setting <<< "$folder"
            python src/model/record_scores.py \
                -data ${OUTPUT_DIR}/eval_results.json \
                -output_path ../outputs/${OUTPUT_FOLDER}/all_score.json \
                -domain ${domain} \
                -task ${system_task} \
                -instruction_type ${instruction_type} \
                -prompt_setting ${prompt_setting}
        done
    done
done

# Aggregate the scores of all the tasks for this model to get the final IHEval score and print it to the console
python src/model/average_final_score.py \
    -record ../outputs/${OUTPUT_FOLDER}/all_score.json \
    -output ../outputs/${OUTPUT_FOLDER}/iheval.jsonl
cd -
