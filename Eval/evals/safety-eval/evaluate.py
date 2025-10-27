# Modify based on https://github.com/paul-rottger/xstest/tree/main

import json
from absl import flags, app

_RESPONSE = flags.DEFINE_string(
    "response", None, "path to response data", required=True
)

_VARIANT = flags.DEFINE_string(
    "variant", None, "variant", required=True
)

def main(argv):
    print(f"{_VARIANT.value} Acc:")
    with open(_RESPONSE.value, "r", encoding="utf-8") as f:
        res = json.load(f)
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app.run(main)