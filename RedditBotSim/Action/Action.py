import sys
# sys.path.append('./Action')
from Basics import Action
from datetime import datetime
from Entity import RedditPost, RedditComment1, RedditUser, SubReddit, RedditComment1
from typing import List, Optional, Callable, Dict, Any, Type
import math



class BrowsePosts(Action):
    def __init__(self):
        super().__init__(
            name="BrowsePosts",
            description="Browse the message flow",
            func=self.browse_recommend,
            input_args_schema={
                "read_start_time": str, 
                "read_end_time": str,
                },
        )

    @staticmethod
    def browse_recommend(action_args, env, agent):
        post= env.get_env_reddit_data().posts 
        comments1 = env.get_env_reddit_data().comments1
        comments2 = env.get_env_reddit_data().comments2
        feed_posts = env.get_env_reddit_data().all_subreddit_recommend(post, comments1, comments2,action_args["read_start_time"], action_args["read_end_time"], agent.subreddit)            
        if feed_posts is None:
            return "No more recommendations."
        else:
            return feed_posts
        
class BrowseComments(Action):
    def __init__(self):
        super().__init__(
            name="BrowseComments",
            description="12344",
            func=self.browse_recommend,
            input_args_schema={
                "read_start_time": str, 
                "read_end_time": str,
                },
        )
    @staticmethod
    def browse_recommend(action_args, env, agent):
        comment= env.get_env_reddit_data().comments        
        feed_comments = env.get_env_reddit_data().all_subreddit_recommend_comments(
            comment, action_args["read_start_time"], action_args["read_end_time"], agent.subreddit)            
        if feed_comments is None:
            return "No more recommendations."
        else:
            return feed_comments              

class PostAction(Action):

    def __init__(self):
        super().__init__(
            name="PostAction",
            description="Publish a post based on the above information.",
            func=self.post,
            input_args_schema={
                "posts": str,
                "time": str,
                "subreddit": str,
            },
        )

    @staticmethod
    def post(action_args: Optional[Dict[str, Any]], env, agent):
        try:
            observation = (
                env.get_env_reddit_data().add_post_by_subreddit_title_content_agent(
                    posts=str(action_args["posts"]),
                    time=action_args["time"],
                    subreddit=action_args["subreddit"],
                    agent=agent,
                )
            )
        except Exception as e:
            print("======Exception====== \n", e)    
            return e
        return observation


class CommentAction1(Action):

    def __init__(self):
        super().__init__(
            name="Comment1",
            description="",
            func=self.comment1,
            input_args_schema={
                "link_id": str,
                "parent_id": str,
                "comment_content": str,
                "time": str,
                "level": str,
                "subreddit": str
            },
        )

    @staticmethod
    def comment1(action_args: Optional[Dict[str, Any]], env, agent):
        try:
            if action_args["link_id"].split("_")[1] in list(env.get_submission_id()):
                observation = env.get_env_reddit_data().add_comment1_by_parent_id_content_agent(
                    link_id=action_args["link_id"],
                    parent_id=action_args["parent_id"],
                    content=str(action_args["comment_content"]),
                    time=action_args["time"],
                    level = action_args["level"],
                    planed_subreddit = action_args["subreddit"],
                    agent=agent,
                )
            else:
                observation = "The link_id is incorrect."
        except Exception as e:
            print("======Exception====== \n", e)    
            return e
        return observation

class CommentAction2(Action):

    def __init__(self):
        super().__init__(
            name="Comment2",
            description="",
            func=self.comment2,
            input_args_schema={
                "link_id": str,
                "parent_id": str,
                "comment_content": str,
                "time": str,
                "level": str
            },
        )

    @staticmethod
    def comment2(action_args: Optional[Dict[str, Any]], env, agent):
        try:
            if action_args["link_id"].split("_")[1] in list(env.get_submission_id()):
                observation = env.get_env_reddit_data().add_comment2_by_parent_id_content_agent(
                    link_id=action_args["link_id"],
                    parent_id=action_args["parent_id"],
                    content=str(action_args["comment_content"]),
                    time=action_args["time"],
                    level = action_args["level"],
                    planed_subreddit = action_args["subreddit"],
                    agent=agent,
                )
            else:
                observation = "The link_id is incorrect."
        except Exception as e:
            print("======Exception====== \n", e)    
            return e
        return observation

class CreateUserAction(Action):
    def __init__(self):
        super().__init__(
            name="CreateUserAction",
            description="",
            func=self.add_agent,
            input_args_schema={
                "user_id": str,
                "name": str,
                "description": str,
                "subreddit": str,
                "submission_num": int,
                "comment_num": int,
                "character_setting":str,
                "comment_num_1": str,
                "comment_num_2": str
            },
        )

    @staticmethod
    def add_agent(action_args: Optional[Dict[str, Any]], env, agent):
        try:
            if action_args["name"] not in list(env.get_user_name()):
                observation = env.get_env_reddit_data().add_user_by_user_name_description(
                    author_id = action_args["user_id"],
                    author_name=action_args["name"],
                    description = action_args["description"],
                    subreddit = action_args["subreddit"],
                    submission_num = action_args["submission_num"],
                    comment_num = action_args["comment_num"],
                    character_setting = action_args["character_setting"],
                    comment_num_1 = action_args["comment_num_1"],
                    comment_num_2 = action_args["comment_num_2"]
                )
            else:
                user_name = action_args["name"]
                reddit_user = RedditUser(
                    author_id = action_args["user_id"],
                    author_name=action_args["name"],
                    description = action_args["description"],
                    subreddit = action_args["subreddit"],
                    submission_num = action_args["submission_num"],
                    comment_num = action_args["comment_num"],
                    character_setting = action_args["character_setting"],
                    comment_num_1 = action_args["comment_num_1"],
                    comment_num_2 = action_args["comment_num_2"]
                )
                observation = (f"The user_name {user_name} is duplicate.", reddit_user)
        except Exception as e:
            print("======Exception====== \n", e)    
            return e
        return observation


class StopAction(Action):
    def __init__(self):
        super().__init__(
            name="Stop",
            description="",
            func=self.stop,
            input_args_schema={},
        )

    @staticmethod
    def stop(action_args: Optional[Dict[str, Any]], env, agent):
        observation = "Something goes wrong, stop this action"
        return observation
