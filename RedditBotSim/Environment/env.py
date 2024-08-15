from RedditData import RedditData
import sys
sys.path.append('./Utils')
from utils import (
    load_config,
    load_action_space,
    get_action_info,
    set_logger,
    load_subreddit_list,
    load_init_observation,
    load_start_time,
    load_end_time,
    load_user_name,
    load_submission_id
)
sys.path.append('./Action')
from Basics import Action
from Action import *
sys.path.append('./Environment')
from Entity import RedditUser
from typing import List, Optional, Callable, Dict, Any, Type


class RedditEnv:

    def __init__(self, config):
        self.config = config
        self.env_task = (
            None  
        )

    def reset(self):
        self.env_reddit_data = RedditData(self.config)
        self.env_action_space = load_action_space(self.config)
        self.env_action_info = get_action_info(self.env_action_space)
        self.subreddit_list = load_subreddit_list(self.config)
        self.user_name = load_user_name(self.config)
        self.submission_id = load_submission_id(self.config)
        # self.logger = set_logger(self.config)
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

    def get_env_subreddit_info(self):
        return self.subreddit_list

    def get_recommend_start_time(self):
        return 

    def get_submission_id(self):
         return self.submission_id

    def get_user_name(self):
        return self.user_name

    def step(self, action: Action, action_args: Dict[str, Any], agent=None):
        if not action.enable:
            action_res = ""
            observation = "Can't do this action now, Please choose another one."
        else:
            action_res = action(action_args, self, agent)
            observation = action_res
        return observation

    def exit(self):
        pass

    def create_user(self, create_action_args):
        return self.step(CreateUserAction(), action_args=create_action_args)
