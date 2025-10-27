# modify base on  
# https://github.com/modelscope/evalscope/blob/main/evalscope/benchmarks/math_500/math_500_adapter.py
# https://github.com/modelscope/evalscope/blob/main/evalscope/metrics/math_parser.py


import json
from absl import flags, app
from utils.math_parser import extract_answer, strip_answer_string, math_equal

_RESPONSE = flags.DEFINE_string(
    "response", None, "path to response data", required=True
)

_VARIANT = flags.DEFINE_string(
    "variant", None, "variant", required=True
)

def main(argv):
    data = []
    with open(_RESPONSE.value, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    
    correct = 0.
    if _VARIANT.value == "math500":
        assert len(data) == 500
    else: assert False

    for i in data:
        pred = strip_answer_string(extract_answer(i['messages'][-1]["content"]))
        gold = strip_answer_string(i['answer'])
        correct += math_equal(pred, gold)
    
    print(f"{_VARIANT.value} Acc: {correct/len(data)}")
    

if __name__ == "__main__":
    app.run(main)

    
