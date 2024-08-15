from collections import defaultdict
from typing import Dict, List
import pandas as pd  
import yaml
import logging
import os
import importlib
import json
import random
import string
from datetime import datetime, timedelta
from math import log
import numpy as np
import re
from ast import literal_eval
import sys
sys.path.append('./Environment')
from Entity import RedditComment1, RedditPost, RedditComment2


def load_create_user_args(config):
    return {
        "user_name": config["user"]["user_name"],
        "icon_img": config["user"]["icon_img"],
        "time": config["user"]["time"],
        "follow_subreddit": config["user"]["follow_subreddit"],
    }


def load_page_size(config):
    return config["constants"]["page_size"]


def load_config(filename):
    with open(filename, "r") as file:
        return yaml.safe_load(file)


def load_subreddit_list(config):
    return config["subreddit"]

def load_user_name(config):
    file_path = config["paths"]["reddit_users"]
    # Read the entire CSV file
    df = pd.read_csv(file_path)  
    # Get data for a column, such as 'column_name'
    user_name = df['name']  
    return user_name

def load_start_time(config):
    return config["time"]["start_time"]

def load_submission_id(config):
    file_path = config["paths"]["reddit_posts"]
    # Read the entire CSV file
    df = pd.read_csv(file_path)  
    # Get data for a column, such as 'column_name'
    submission_id = df['submission_id']  
    return submission_id

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
        action_space[action_name] = action_class()
    return action_space


def get_action_info(action_space):
    action_info_list = []
    for action_name, action_obj in action_space.items():
        action_info = {
            "ActionName": action_name,
            "ActionArgs": list(action_obj.input_args_schema.keys()),
            "Description": action_obj.description.strip(),
        }
        action_info_list.append(action_info)
    return action_info_list


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
    handler = logging.FileHandler(log_file, mode="a", encoding='utf-8')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger



def epoch_seconds(date):
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    td = date - datetime(1970, 1, 1)
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


def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1.T, embedding2)
    norm_embedding1 = np.linalg.norm(embedding1)
    norm_embedding2 = np.linalg.norm(embedding2)
    cosine_sim = dot_product / (norm_embedding1 * norm_embedding2)
    return cosine_sim


def put_comments_to_post(
    post_dict: Dict[str, RedditPost], comments1: List[RedditComment1], comments2: List[RedditComment2]
):
    submiss_comments1 = defaultdict(list)
    # select comment 1 and comment 2
    submiss_comments2 = defaultdict(list)
    # Group all comments by link_id
    for comment1 in comments1:
        if comment1.link_id != "":
            submiss_comments1[comment1.link_id.split("_")[1]].append(comment1)


    for comment2 in comments2:
        if comment2.link_id != "":
            submiss_comments2[comment2.link_id.split("_")[1]].append(comment2)

    # Put comments in the corresponding dictionary
    for submiss_id, comment1 in submiss_comments1.items():
        post_dict[submiss_id].comments1 = comment1

    for submiss_id, comment2 in submiss_comments2.items():
        post_dict[submiss_id].comments2 = comment2

    return post_dict





def random_date(start, end):
    """
    Generates a random time between a given start and end time.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)


def generate_input_args(action_args_schema, start_time_str, end_time_str):
    # Parse start and end times
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")

    input_args = {}
    for arg_name, arg_type in action_args_schema:
        if arg_type == str:
            if arg_name == "time":
                # Generates a time within the specified range
                random_time = random_date(start_time, end_time)
                input_args[arg_name] = random_time.strftime("%Y-%m-%d %H:%M:%S")
            else:
                input_args[arg_name] = f"{{ {arg_name} }}"
        elif arg_type == bool:
            input_args[arg_name] = True  # Or False
        elif arg_type == int:
            input_args[arg_name] = 0  # Or other value
        # Add other types of processing logic
    return input_args


def initialize_observation(start_time_str, end_time_str, subreddit_list, action_info):
    # Execution time range
    start_time = datetime.strptime(
        start_time_str, "%Y-%m-%d %H:%M:%S"
    )  # Get the start time from the environment
    end_time = datetime.strptime(
        end_time_str, "%Y-%m-%d %H:%M:%S"
    )  # Gets the end time from the environment
    # Execution start time
    current_time = random_date(start_time, end_time)

    # Extract the list of subreddits
    subreddits = subreddit_list  # Get subreddits from the environment

    # Extract all action names from action_space
    actions = [action["ActionName"] for action in action_info]

    # Initializes the specific structure of user information, posts, and comments as an empty list in dictionary format
    users_ob = []  # The elements in the list are RedditUser dictionaries
    posts_ob = []  # The elements in the list are RedditPost dictionaries
    comments_ob = []  # The elements in the list are dictionaries in RedditComment format

    # Building observation dictionary
    observation = {
        "start_time": start_time,
        "end_time": end_time,
        "current_time": current_time,
        "subreddits": subreddits,
        "action_space": actions,
        "users_ob": users_ob,
        "posts_ob": posts_ob,
        "comments_ob": comments_ob,
    }

    return observation
