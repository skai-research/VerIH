import re
import os
import json
import tqdm
import random
import traceback
import numpy as np
from llm import formate_json, OpenAIInterface
from datasets import load_dataset
from collections import Counter
random.seed(42)

def divide_prompt(prompt, pattern, gt_str):
    gt = json.loads(gt_str)
    del gt["func_name"]

    if gt["keyword_list"]:
        key_words = gt["keyword_list"]
        assert len(key_words) == 2
        gt["keyword1"] = key_words[0]
        gt["keyword2"] = key_words[1]
        del gt["keyword_list"]

    if gt["forbidden_words"]:
        gt["forbidden_words"] = ", ".join(gt["forbidden_words"]).strip()

    renamed_gt = {k.replace("_", " "): v for k, v in gt.items()}
    filtered_gt = {k: v for k, v in renamed_gt.items() if v is not None}   

    if "quantifier" in filtered_gt:
        q = filtered_gt["quantifier"]
        pattern = pattern.replace("at least / around / at most", q)
    constraint = pattern.format(**filtered_gt).strip()

    assert constraint in prompt
    rest = prompt.replace(constraint, "").strip()
    return constraint, rest


def rewrite_prompt(num_samples, skip=False):
    model = OpenAIInterface(model="gpt-4o")
    # model = OpenAIInterface(model="Anthropic Claude 4 Sonnet")
    raw_dataset = load_dataset("allenai/RLVR-IFeval")["train"]
    name_counts = Counter(item["constraint_type"] for item in raw_dataset)

    samples = {}
    sample_per_group = num_samples // len(name_counts)
    for dtype in name_counts.keys():
        group = raw_dataset.filter(lambda x: x['constraint_type'] == dtype, desc="")
        print(f"sample from {dtype} {sample_per_group}/{len(group)}")
        group = group.shuffle(seed=42).select(range(sample_per_group))
        samples[dtype] = group
    print(f"Total Samples {sum(len(x) for x in samples.values())}")

    system_request = '''One task consists of two parts: a constraint and a detailed instruction. Together, they define the task and there is no conflict between them.
Your job is to rewrite the detailed instruction so that there will be a conflict between them. 
Keep the intention, style, and language in original instruction.
The modify should be subtle and difficult to detect the conflict with system prompts.
Please also explain how you modify the instruction in the 'explain' part within 20 words.

Return Format: a json object that contain the prompt, example: {"instruction":"xxxx", "explain":"xxxx"}'''
    user_template = '''Constrain \n{} \nInstruction \n{}'''
    dataset = []
    for k, v in samples.items():
        for i in tqdm.tqdm(v):        
            # Read from Raw Dataset
            msg = i["messages"]
            assert len(msg) == 1
            sys_prompt, user_prompt = divide_prompt(msg[0]["content"], i["constraint"].strip(), i["ground_truth"])

            # Change Prompts
            aligned = random.choices([0, 1], weights=[0.6, 0.4])[0] # conflict rewrite may fail. So try more
            if (not aligned) and (not skip):
                try:
                    response = model.generate(system_request, user_template.format(sys_prompt, user_prompt), formate_json)
                    if (response is None) or ("instruction" not in response) or ("explain" not in response): raise Exception("Generation Error")
                    else: 
                        user_prompt = response["instruction"]
                        explain = response["explain"]
                except Exception as _:
                    traceback.print_exc()
                    print(json.dumps(i, indent=4, ensure_ascii=False))
            else:
                explain = ""

            dataset.append({
                "sys_prompt": sys_prompt,
                "user_prompt": user_prompt,
                "explain": explain,
                "raw_messages": msg,
                "gt": i["ground_truth"],
                "type": f"{k.lower()}:{'aligned' if aligned else 'conflict'}",
            })
    return dataset

def save_dataset(dataset):
    random.shuffle(dataset)
    train_ratio = 0.9
    train_size = int(len(dataset) * train_ratio)
    train_split = dataset[:train_size]
    test_split = dataset[train_size:]

    save_dir = "dataset/verih"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"train.json"), "w", encoding="utf-8") as f:
        json.dump(train_split, f, indent=4, ensure_ascii=False)

    with open(os.path.join(save_dir, f"test.json"), "w", encoding="utf-8") as f:
        json.dump(test_split, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    dataset = rewrite_prompt(8000)
    print(f"Total Generated {len(dataset)}")
    save_dataset(dataset)
    print(f"API Cost {OpenAIInterface.total_cost}")