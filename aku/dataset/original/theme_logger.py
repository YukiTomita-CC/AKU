import json
import random


def get_random_theme_and_persona():
    txt_file_path = "docs/talk_theme.txt"
    json_file_path = "docs/user_persona.json"
    log_file_path = "aku/dataset/original/assets/log.txt"

    with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
        themes = [line.strip() for line in txt_file if line.strip()]

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        user_personas = json.load(json_file)

    log = []
    try:
        with open(log_file_path, 'r', encoding='utf-8') as log_file:
            for line in log_file:
                log.append(json.loads(line.strip()))
    except FileNotFoundError:
        pass

    while True:
        theme = random.choice(themes)
        user_persona = random.choice(user_personas)
        persona_id = user_persona["id"]

        combination = {"theme": theme, "persona_id": persona_id}
        if combination not in log:
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(json.dumps(combination, ensure_ascii=False) + '\n')

            return {"theme": theme, "user_persona": user_persona}
