"""Chat templating and tokenization for conversations stored in an OpenAI-ish format. Supports system, user, assistant, and tool roles.

Expected input format:

```
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like today?"},
        {"role": "assistant", "content": "Where are you?"},
        {"role": "user", "content": "I am in New York"},
        {"role": "assistant", "content": "[{\\"id\\": \\"123456789\\", \\"type\\": \\"function\\", \\"function\\": {\\"name\\": \\"get_current_weather\\", \\"arguments\\": \\"{\\\\\\"location\\\\\\": \\\\\\"New York\\\\\\"}\\"}}]"},
        {"role": "tool", "content": "{\"status\": \"success\", \"temperature\": \"75 degrees\", \"wind\": \"10 mph\", \"humidity\": \"50%\"}", "tool_call_id": "123456789"},
        {"role": "assistant", "content": "The weather in New York is 75 degrees with a wind of 10 mph and a humidity of 50%."},
    ],
}
```

Tool responses can be arbitrarily formatted strings. We recommend JSON-encoding the OpenAI tool call objects.

We delegate formatting/handling of tool calls to the user. Any tool descriptions should be placed in the system message.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    OTHER = "other"

    def __int__(self):
        if self == Role.SYSTEM:
            return 0
        elif self == Role.USER:
            return 1
        elif self == Role.ASSISTANT:
            return 2
        elif self == Role.TOOL:
            return 3
        else:
            return 4


class Message(BaseModel):
    role: Role
    content: str = ""


class Conversation(BaseModel):
    messages: List[Message]
    tools: Optional[List[Dict]] = None


class TemplateConfig(BaseModel):
    """
    Configuration for tokenization. Only a subset of these must be non-empty.
    """

    bos: List[int] = []
    system_start: List[int] = []
    system_end: List[int] = []
    user_start: List[int] = []
    user_end: List[int] = []
    assistant_start: List[int] = []
    assistant_end: List[int] = []
    tool_start: List[int] = []
    tool_end: List[int] = []

    # settings
    strip_whitespace: bool = False


def tokenize_conversation(
    conversation: Union[Conversation, Dict, List[Dict]],
    tokenizer: PreTrainedTokenizerBase,
    config: TemplateConfig,
    add_generation_prompt: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Tokenizes a conversation and returns a list of tokens, role ids, and a mask indicating which tokens should be trained on.

    Args:
        conversation: Conversation object, dictionary, or list of messages.
        tokenizer: A PreTrainedTokenizerBase object
        config: A TokenizerConfig object
        add_generation_prompt: Whether to append a generation prompt (assistant start tokens)
    """
    out = []
    role_ids = []

    if isinstance(conversation, Dict):
        conversation = Conversation(**conversation)
    elif isinstance(conversation, List):
        conversation = Conversation(messages=conversation)

    if config.bos:
        out.extend(config.bos)
        role_ids.extend([int(Role.OTHER)] * len(config.bos))

    for message in conversation.messages:
        content = message.content
        if config.strip_whitespace:
            content = content.strip()
        content_tokens = tokenizer.encode(content, add_special_tokens=False)

        if message.role == Role.SYSTEM:
            out.extend(config.system_start)
            role_ids.extend([int(Role.OTHER)] * len(config.system_start))
            out.extend(content_tokens)
            out.extend(config.system_end)
            role_ids.extend([int(Role.SYSTEM)] * (len(out) - len(role_ids)))

        elif message.role == Role.USER:
            out.extend(config.user_start)
            role_ids.extend([int(Role.OTHER)] * len(config.user_start))
            out.extend(content_tokens)
            out.extend(config.user_end)
            role_ids.extend([int(Role.USER)] * (len(out) - len(role_ids)))

        elif message.role == Role.ASSISTANT:
            out.extend(config.assistant_start)
            role_ids.extend([int(Role.OTHER)] * len(config.assistant_start))
            out.extend(content_tokens)
            out.extend(config.assistant_end)
            role_ids.extend([int(Role.ASSISTANT)] * (len(out) - len(role_ids)))

        elif message.role == Role.TOOL:
            out.extend(config.tool_start)
            role_ids.extend([int(Role.OTHER)] * len(config.tool_start))
            out.extend(content_tokens)
            out.extend(config.tool_end)
            role_ids.extend([int(Role.TOOL)] * (len(out) - len(role_ids)))

        else:
            raise ValueError(f"Invalid role: {message.role}")

    if add_generation_prompt:
        assert conversation.messages[-1].role in [Role.USER, Role.TOOL]
        out.extend(config.assistant_start)
        role_ids.extend([int(Role.OTHER)] * len(config.assistant_start))

    assert len(out) == len(role_ids)

    return out, role_ids
