import json
from absl import flags, app

_RESPONSE = flags.DEFINE_string(
    "response", None, "path to response data", required=True
)

_VARIANT = flags.DEFINE_string(
    "variant", None, "variant", required=True
)

def main(argv):
    with open(_RESPONSE.value, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(len(data)) # assert len(data) == 10
    print(f"{_VARIANT.value} aligned: {data['overall']['aligned']}")
    print(f"{_VARIANT.value} conflict: {data['overall']['conflict']}")
    

if __name__ == "__main__":
    app.run(main)