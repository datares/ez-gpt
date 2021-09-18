import json


def read_config():
    with open("config.json") as f:
        return json.load(f)


config = read_config()
