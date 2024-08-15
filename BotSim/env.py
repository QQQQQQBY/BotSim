from data_operations import RedditData
from utils import (
    load_config,
    load_action_space,
    get_action_info,
    set_logger,
    load_subreddit_list,
    load_init_observation,
    load_start_time,
    load_end_time,
)
from basic import Action
from action import *
from entity import RedditUser
from typing import List, Optional, Callable, Dict, Any, Type


class RedditEnv:

    def __init__(self, config):
        self.config = config

    def reset(self):
        self.env_reddit_data = RedditData(self.config)
        self.env_action_space = load_action_space(self.config)
        self.env_action_info = get_action_info(self.env_action_space)
        self.subreddit_list = load_subreddit_list(self.config)
        self.logger = set_logger(self.config)
        self.start_time = load_start_time(self.config)
        self.end_time = load_end_time(self.config)
        init_observation = load_init_observation(
            env_action_info=self.env_action_info,
            subreddit_list=self.subreddit_list,
            start_time=self.start_time,
            end_time=self.end_time,
        )
        return init_observation

    def get_env_reddit_data(self):
        return self.env_reddit_data

    def get_env_action_space(self):
        return self.env_action_space

    def get_env_action_info(self):
        return self.env_action_info

    def step(self, action: Action, action_args: Dict[str, Any], agent=None):
        if not action.enable:
            action_res = ""
            observation = "Can't do this action now, Please choose another one."
        else:
            action_res = action(action_args, self, agent)
            observation = action_res
            if agent:
                self.logger.info(
                    f"agent id: {agent.author_id}, action name: {action.name}, action args: {action_args}, result: {observation}"
                )
            else:
                self.logger.info(
                    f"action name: {action.name}, action args: {action_args}, result: {observation}"
                )
        return observation

    def exit(self):
        pass
