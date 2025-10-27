# Generation
### run_vllm.sh
    python utils/evaluation/eval.py generators generators --use_vllm

### evaluation/eval.py
    generators()
    {
        load_generative_model()
        evaluation_tasks: list[OpenEndedTaskBase] = load_evaluation_tasks(EvalMode.GENERATION, tasks)
        eval_task.validate_if_model_is_compatible(model)
        report, individual_results = eval_task.evaluate(model)
    }

### evaluation/models/__init__.py
    load_generative_model()
    {
        VllmGeneratorModel()
    }

### evaluation/schemas.py
    class OpenEndedTaskBase(TaskBase)
    {
        validate_if_model_is_compatible()
        _evaluate()
    }

    class TaskBase():
    {
        self.data = self.put_id_on_data(self.load())
    }

    class VllmGeneratorModel()
    {
        generate_completions(formatted_inputs)
        {
            formatted_inputs = [self.model_input_template.format(**d) for d in inputs]
            subprocess_inference_with_vllm()
        }
    }

### src/generation_utils.py
    subprocess_inference_with_vllm()
    {
        create_and_inference_with_vllm()
    }

    create_and_inference_with_vllm()
    {
        llm = load_vllm_model(model_name_or_path, tokenizer_mode=tokenizer_mode)
        result = inference_with_vllm()
    }

    inference_with_vllm()
    {
        # Call vlllm/torchllms
    }

# Tasks
### evaluation/tasks/generation/base.py
    _evaluate()
    {
        completions = model.generate_completions(self.data, max_new_tokens=self.max_new_tokens)
    }

### evaluation/tasks/generation/wildjailbreak/__init__.py
    base._evaluate()
    load("benign/harmful.jsonl") -> self.data

### evaluation/tasks/generation/harmbench/__init__.py    
    {
        inputs = [{"instruction": row["Behavior"]} for row in self.data]
        completions = model.generate_completions(inputs)
        evaluation_outputs = classifier.classify(evaluator_inputs) # WildGuard
    }

    load("harmbench_behaviors_text_test.csv") -> self.data

### src/classifier_models/wildguard.py
    classify()
    {
        _classify_batch()
        {
            subprocess_inference_with_vllm()
        }
    }

# Others
### evaluation/tasks/generation/xstest/__init__.py
    {None}

### evaluation/tasks/generation/mmlu/__init__.py
    {
        model.model_input_template = current_model_input_template + "The answer is:"
        predictions, probs = model.get_next_word_predictions(hf_model, hf_tokenizer, prompts, choices)
    }

