import yaml
import logging
import os
import importlib
import json
import random
import string
from datetime import datetime
from math import log


def load_config(filename):
    with open(filename, "r") as file:
        return yaml.safe_load(file)


def load_subreddit_list(config):
    return config["subreddit"]


def load_start_time(config):
    return config["time"]["start_time"]


def load_end_time(config):
    return config["time"]["end_time"]


def check_time(cur_time: datetime, config):
    start_time = str_to_datetime(load_start_time(config))
    end_time = str_to_datetime(load_end_time(config))
    return start_time <= cur_time <= end_time


def load_init_observation(env_action_info, subreddit_list, start_time, end_time):
    return f"""
        System Overview:
        Welcome to the Reddit Simulation System! This system aims to replicate the functionalities and interactions observed on the Reddit platform. Users can engage in various actions.
        System Time:
        System Start Time: {start_time}, System End Time: {end_time}
        Action Space:
        The system provides the following actions for intelligent agents to execute: {env_action_info}.
        Subreddit List:
        The system contains the following subreddits: {subreddit_list}.
        The users can select actions and specify parameters. The system will simulate user interactions based on these actions, aiming to emulate real Reddit behavior.
        The users must conduct their actions within the specified time frame, which ranges from the system's start time to its end time. It is imperative for agents to ensure that their actions fall within this time period.
    """


def str_to_datetime(time_str: str) -> datetime:
    return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")


def load_action_space(config):
    action_space = {}
    for action_name, action_path in config["actions"].items():
        module_name, class_name = action_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        action_class = getattr(module, class_name)
        input_args_schema = getattr(action_class(), "input_args_schema", None)
        description = getattr(action_class(), "description", None)
        action_space[action_name] = (action_class, input_args_schema, description)

    return action_space


def get_action_info(action_space):
    return [
        {
            "ActionName": action_name,
            "ActionArgs": list(input_args_schema.keys()),
            "Description": description,
        }
        for action_name, (
            action_class,
            input_args_schema,
            description,
        ) in action_space.items()
    ]


def set_logger(config, name="default"):
    log_file = config["log"]["log_file"]
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    log_folder = os.path.join(output_folder, "log")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    log_file = os.path.join(log_folder, log_file)
    handler = logging.FileHandler(log_file, mode="a")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger


def epoch_seconds(date):
    date_time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    td = date_time - datetime(1970, 1, 1)
    return td.days * 86400 + td.seconds + (float(td.microseconds) / 1000000)


def hot(ups, downs, date):
    s = ups - downs
    order = log(max(abs(s), 1), 10)
    sign = 1 if s > 0 else -1 if s < 0 else 0
    seconds = epoch_seconds(date) - 1134028003
    return round(order + sign * seconds / 45000, 7)


def hot_by_score_time(s, date):
    """The hot formula. Should match the equivalent function in postgres."""
    order = log(max(abs(s), 1), 10)
    sign = 1 if s > 0 else -1 if s < 0 else 0
    seconds = epoch_seconds(date) - 1134028003
    return round(order + sign * seconds / 45000, 7)


def list2str(input_list):
    return json.dumps(input_list)


def format_time(time: datetime):
    return time.strftime("%Y-%m-%d %H:%M:%S")


def generate_reddit_id(existing_ids, length=7, max_retries=1000):
    characters = string.ascii_lowercase + string.digits
    existing_ids_set = set(existing_ids)

    for _ in range(max_retries):
        reddit_id = "".join(random.choice(characters) for _ in range(length))
        if reddit_id not in existing_ids_set:
            return reddit_id

    raise ValueError("Unable to generate a unique Reddit ID.")


# print(epoch)
# time = datetime.now()
# print(hot(10,0,time))
# config = load_config("./config.yaml")

# action_space = load_action_space(config)
# print(get_action_info(action_space))
