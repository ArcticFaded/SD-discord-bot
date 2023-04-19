import json
from pathlib import Path
from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards import WildcardManager
import re


config = json.load(open("config.json"))

wm = WildcardManager(Path(config["prompts"]["wildcard_path"]))
generator = RandomPromptGenerator(wildcard_manager=wm)



def get_index():
    return json.load(open(Path(config["prompts"]["index_path"])))
    
def replace_wildcards(text):
    index = get_index()
    # Sort the keys in index by length in descending order
    keys = sorted(index.keys(), key=lambda x: len(x), reverse=True)

    # Create a pattern to search for the keys in the text
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(k) for k in keys) + r')\b')

    # Replace the found keys with their corresponding values in the index
    result = pattern.sub(lambda x: index[x.group(0)], text)

    return result


def fluff_prompt(prompt, test_restore=False):
    wild_prompt = replace_wildcards(prompt)
    
    restore_face = False
    if test_restore:
        for i in ["a1", "a2", "b1", "b2"]:
            if i in wild_prompt:
                restore_face = True
                break

    return generator.generate(wild_prompt, 1)[0], restore_face
