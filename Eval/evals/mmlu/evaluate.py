# modify base on  
# https://github.com/modelscope/evalscope/blob/main/evalscope/benchmarks/mmlu/mmlu_adapter.py
# https://github.com/modelscope/evalscope/blob/main/evalscope/metrics/completion_parsers.py

import re
import json
from absl import flags, app

_RESPONSE = flags.DEFINE_string(
    "response", None, "path to response data", required=True
)

_VARIANT = flags.DEFINE_string(
    "variant", None, "variant", required=True
)

def process_options(options: list[str]) -> str:
    # Escape each option to ensure special characters in options are treated literally
    escaped_options = [re.escape(option) for option in options]
    # Join options into a regex pattern separated by '|', to match any of the options
    options_pattern = '|'.join(escaped_options)
    return options_pattern

def parse_last_capital(text: str, options: list[str]) -> str:
    for t in text[::-1]:
        if t.isupper() and (t in options):
            return t
    return ''

def parse_first_option(text: str, options: list[str]) -> str:
    options_pattern = process_options(options)

    patterns = [
        rf'[Aa]nswer:\s*({options_pattern})',
        rf'ANSWER:\s*({options_pattern})',
        rf'answer is \(?({options_pattern})\)?',
        rf'[Tt]he correct answer is:\s*({options_pattern})',
        rf'[Tt]he correct answer is:\n\s*({options_pattern})',
        rf'[Tt]he correct answer is:\n\n-\s*({options_pattern})',
        rf'[Tt]he answer might be:\n\n-\s*({options_pattern})',
        rf'[Tt]he answer is \s*({options_pattern})',
    ]

    regexes = [re.compile(pattern) for pattern in patterns]
    for regex in regexes:
        matches = regex.search(text)
        if matches:
            return matches.group(1)
    # If no match found, try to find the last capital letter in the text
    last_capital = parse_last_capital(text, options)
    if last_capital:
        return last_capital
    return 'No valid option found'

def exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0
    return 1 if gold.strip() == pred.strip() else 0

def main(argv):
    options = ['A', 'B', 'C', 'D']
    data = []
    with open(_RESPONSE.value, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
  
    correct = 0.
    for i in data:
        pred = i['messages'][-1]["content"].strip()
        if (len(pred) == 1) and (pred in options):
            pass
        else:
            pred = parse_first_option(pred, options=options)

        gold = i['answer']
        correct += exact_match(gold=gold, pred=pred)
    
    print(f"{_VARIANT.value} Acc: {correct/len(data)}")
    

if __name__ == "__main__":
    app.run(main)
