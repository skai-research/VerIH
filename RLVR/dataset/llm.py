import os
import litellm
import traceback
from openai import OpenAI


class OpenAIInterface(object):
    total_cost = 0
    def __init__(self, model="gpt-4o", temperature=0.9):
        self.model = model
        self.temperature = temperature
        self.max_new_tokens=4096
        self.completion_func = self._offical_api

    def generate(self, system_prompt, user_prompt, formater=None, default_value=None):
        prompt_input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self._generate(prompt_input, formater, default_value)

    def _generate(self, prompt_input, formater, default_value):
        # Retry 3 times if LLM error or Format error
        for i in range(3):
            msg = None
            try:
                response = self.completion_func(prompt_input)
                cost = litellm.completion_cost(completion_response=response)
                self.__class__.total_cost += cost

                # Results
                msg = self._extract_msg(response)
                if formater: return formater(msg)
                else: return msg
            except Exception as e:
                traceback.print_exc()
                print(msg)
        return default_value

    def _extract_msg(self, response):
        return response.choices[0].message.content

    def _offical_api(self, prompt_input):
        response = litellm.completion(
            model=self.model,
            messages=prompt_input,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature
        )
        return response


import json
import re
def formate_json(msg):
    # Filter outer text by regex
    pattern = r"```(.*?)```"
    match = re.search(pattern, msg, re.DOTALL)
    if match:
        msg = match.group(1)

    # Construct json object
    msg = msg.strip().removeprefix("json").strip()
    res = json.loads(msg)
    return res
