import csv
import sys
sys.path.append('./RedditBotSim/Environment')
from Entity import RedditPost, RedditComment1, RedditComment2, RedditUser, SubReddit
from datetime import datetime
sys.path.append('./RedditBotSim/Utils')
from utils import (
    load_config,
    hot_by_score_time,
    list2str,
    generate_reddit_id,
    format_time,
    check_time,
    load_start_time,
    load_end_time,
    str_to_datetime,
)
from typing import List, Dict, Any
import json
import time
from utils import *
# from sentence_transformers import SentenceTransformer
from itertools import product
import pandas as pd
import ast
class RedditData:
    """
    Data class.
    """

    def __init__(self, config):
        self.config = config
        self.posts: Dict[str, RedditPost] = {}
        self.comments1: List[RedditComment1] = []
        self.comments2: List[RedditComment2] = []
        self.users: Dict[str, RedditUser] = {}
        self.name_submission_id_dict: Dict[str, str] = {}
        self.page_size: int = self.config["constants"]["page_size"]
        self._load_data(config)

    # recommend posts and comments
    def all_subreddit_recommend(
        self, post: Dict[str, RedditPost], comments1: Dict[str, RedditComment1], comments2: Dict[str, RedditComment2], start_date: str, end_date: str, subreddit: str
    ):
        try:
            subreddit = ast.literal_eval(subreddit)
            start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
            end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
            filtered_post_dict = {key: value
                for key, value in post.items()
                if value.subreddit in subreddit}
            filtered_post_dict = {
                key: value
                for key, value in filtered_post_dict.items()
                if datetime.strptime(str(value.time), "%Y-%m-%d %H:%M:%S") < end_date and datetime.strptime(str(value.time), "%Y-%m-%d %H:%M:%S") > start_date
            }
            posts_id_list = list(filtered_post_dict.keys())

            if len(filtered_post_dict) ==0:
                return "empty"
            filter_comments1_by_time = [
                value_comment1
                for value_comment1 in comments1
                if datetime.strptime(str(value_comment1.comment1_time), "%Y-%m-%d %H:%M:%S") < end_date and datetime.strptime(str(value_comment1.comment1_time), "%Y-%m-%d %H:%M:%S") > start_date
            ]

            filter_comments1_by_time = [
                value_comment1
                for value_comment1 in filter_comments1_by_time
                if value_comment1.link_id.split("_")[1] in  posts_id_list
            ]

            filter_comments2_by_time = [
                value_comment2
                for value_comment2 in comments2
                if datetime.strptime(str(value_comment2.comment2_time), "%Y-%m-%d %H:%M:%S") < end_date and datetime.strptime(str(value_comment2.comment2_time), "%Y-%m-%d %H:%M:%S") > start_date
            ]

            filter_comments2_by_time = [
                value_comment2
                for value_comment2 in filter_comments2_by_time
                if value_comment2.link_id.split("_")[1] in  posts_id_list
            ]

            for key in filtered_post_dict:
                ups = filtered_post_dict[key].score
                if filtered_post_dict[key].upvote_ratio != 0:
                    downs = (filtered_post_dict[key].score / filtered_post_dict[key].upvote_ratio) * (1 - filtered_post_dict[key].upvote_ratio)
                else:
                    downs = 0
                hot_score = hot(
                    ups, downs, filtered_post_dict[key].time)
                filtered_post_dict[key].hot_score = hot_score
            # Sort the post by hot and date
            sorted_post_dict = dict(
                sorted(
                    filtered_post_dict.items(),
                    key=lambda x: (
                        -x[1].hot_score,
                        -datetime.strptime(str(x[1].time), "%Y-%m-%d %H:%M:%S").timestamp(),
                    ),
                    reverse=False,
                )
            )


            filtered_post_comment_dict = put_comments_to_post(
                sorted_post_dict, filter_comments1_by_time, filter_comments2_by_time
            )
            return filtered_post_comment_dict
        except Exception as e:
            print("======Exception====== /n", e) 
            return []



    def add_post_by_subreddit_title_content_agent(
        self,
        posts: str,
        time: str,
        subreddit: str,
        agent: RedditUser,
    ):
        time = str_to_datetime(time_str=time)
        if not check_time(cur_time=time, config=self.config):
            start_time = load_start_time(config=self.config)
            end_time = load_end_time(config=self.config)
            return f"Input time is {time}. Start time is {start_time}, end time is {end_time}. The User must conduct their actions within the specified time frame."
        existing_post_id = list(self.get_posts().keys())
        submission_id = generate_reddit_id(existing_post_id)
        add_post = RedditPost(
            submission_id=submission_id,
            author_id=agent.author_id,
            author_name=agent.author_name,
            posts=posts,
            score=0,
            num_comments=0,
            upvote_ratio=0.0,
            time=time,
            subreddit=subreddit,
        )
        self.posts[submission_id] = add_post
        agent.add_user_post_by_post(add_post)
        add_post.save_to_file(filename=self.config["paths"]["reddit_posts"])
        return f"Post Success! The Post ID is {submission_id}"

    def add_user_by_user_name_description(
        self, author_id: str, author_name: str, description: str, subreddit: str, submission_num: int, comment_num: int, character_setting: str, comment_num_1: str, comment_num_2: str
    ):
        
        add_user = RedditUser(
            author_id=author_id,
            author_name=author_name,
            description = description,           
            submission_num = submission_num,
            comment_num = comment_num,
            character_setting = character_setting,
            comment_num_1 = comment_num_1,
            comment_num_2 = comment_num_2,
            subreddit = subreddit
        )
        self.users[author_id] = add_user
        add_user.save_to_file(filename=self.config["paths"]["reddit_users"])
        return (f"Create User Sucess! Your User ID is {author_id}", add_user)

    def add_comment1_by_parent_id_content_agent(
        self, link_id: str, parent_id: str, content: str, time: str, level: str,planed_subreddit: str, agent: RedditUser
    ):

        # time = str_to_datetime(time_str=time)
        if not check_time(cur_time=str_to_datetime(time_str=time), config=self.config):
            start_time = load_start_time(config=self.config)
            end_time = load_end_time(config=self.config)
            return f"Input time is {time}. Start time is {start_time}, end time is {end_time}. The User must conduct their actions within the specified time frame."
        submission_id = link_id.split("_")[1]
        if agent == None:
            return "User Not Exist!"

        # Get all comment_id
        existing_comment1_id = []
        for comment in self.comments1:
            existing_comment1_id.append(comment.comment1_id)
        new_comment1_id = generate_reddit_id(existing_comment1_id)


        add_exe_comment1 = RedditComment1(
                comment1_id=new_comment1_id,  
                comment1_author_id=agent.author_id,
                comment1_author_name=agent.author_name,                
                comment1_score=0,
                comment1_body=content,                             
                link_id=link_id,
                parent_id=parent_id,
                subreddit=planed_subreddit,
                comment1_time=time,
                level = level
            )        
        # self.posts[submission_id].add_comment(add_exe_comment)
        self.comments1.append(add_exe_comment1)
        # agent.add_user_comment_by_commet(add_exe_comment)
        add_exe_comment1.save_to_file(
            filename=self.config["paths"]["reddit_comments1"]
        )
        return "Comment Success!"
        # return "Comment Post Not Exist!"
    
    def add_comment2_by_parent_id_content_agent(
        self, link_id: str, parent_id: str, content: str, time: str, level: str,planed_subreddit: str, agent: RedditUser
    ):

        # time = str_to_datetime(time_str=time)
        if not check_time(cur_time=str_to_datetime(time_str=time), config=self.config):
            start_time = load_start_time(config=self.config)
            end_time = load_end_time(config=self.config)
            return f"Input time is {time}. Start time is {start_time}, end time is {end_time}. The User must conduct their actions within the specified time frame."
        submission_id = link_id.split("_")[1]
        if agent == None:
            return "User Not Exist!"

        # Get all comment_id
        existing_comment2_id = []
        for comment in self.comments2:
            existing_comment2_id.append(comment.comment2_id)
        new_comment2_id = generate_reddit_id(existing_comment2_id)

        add_exe_comment2 = RedditComment2(
                comment2_id=new_comment2_id,  
                comment2_author_id=agent.author_id,
                comment2_author_name=agent.author_name,                
                comment2_score=0,
                comment2_body=content,                             
                link_id=link_id,
                parent_id=parent_id,
                subreddit=planed_subreddit,
                comment2_time=time,
                level = level
            )        
        # self.posts[submission_id].add_comment(add_exe_comment)
        self.comments2.append(add_exe_comment2)
        # agent.add_user_comment_by_commet(add_exe_comment)
        add_exe_comment2.save_to_file(
            filename=self.config["paths"]["reddit_comments2"]
        )
        return "Comment Success!"

    def user_add_follow_subreddit(
        self,
        follow_subreddit: str,
        SubReddit_names: List,
        all_subreddit_list: List[SubReddit],
        agent: RedditUser,
    ):
        if follow_subreddit in SubReddit_names:
            subreddit = all_subreddit_list[follow_subreddit]
            agent.add_follow_subreddit(subreddit)
            agent.save_follow_subreddit_to_file(
                filename=self.config["paths"]["follow_subreddit"],
                follow_subreddit=subreddit,
            )
            return f"Follow {follow_subreddit} Success!"
        else:
            return "Follow subreddit Not Exists!"

    def like_post_by_post_id(self, post_id: str):
        if post_id in self.posts.keys():
            self.posts[post_id].score_add_one()
            self.posts[post_id].add_upvote_ratio()
            return "Like Post Success!"
        return "Like Post Not Exist!"

    def dislike_post_by_post_id(self, post_id: str):
        if post_id in self.posts.keys():
            self.posts[post_id].score_minus_one()
            self.posts[post_id].minus_upvote_ratio()
            return "Dislike Post Success!"
        return "Dislike Post Not Exist!"

    def search_post_by_keywords(self, keywords: str):
        matched_posts = []
        keywords_list = keywords.split()
        for post in self.posts.values():
            for keyword in keywords_list:
                if keyword in post.posts or keyword in post.selftext:
                    matched_posts.append(post)
                    break
        if len(matched_posts) == 0:
            return "There Is No Post Matched The Keywords!"
        return self._get_post_content_by_post_list(matched_posts)

    def get_search_post_by_keywords_pageid(self, keywords: str, page_id: int):
        matched_posts = []
        keywords_list = keywords.split()
        for post in self.posts.values():
            for keyword in keywords_list:
                if keyword in post.posts or keyword in post.selftext:
                    matched_posts.append(post)
                    break
        if len(matched_posts) == 0:
            return "There Is No Post Matched The Keywords!"
        post_list_by_search_page = self._get_post_by_page_id_post_list(
            page_id, matched_posts
        )
        if post_list_by_search_page:
            return self._get_post_content_by_post_list(post_list_by_search_page)
        else:
            return "The page id is out of limitation"

    def search_user_by_keywords(self, keywords: str):
        matched_users = []
        keywords_list = keywords.split()
        for user in self.users.values():
            for keyword in keywords_list:
                if keyword in user.author_name or keyword in user.author_id:
                    matched_users.append(user)
                    break
        if len(matched_users) == 0:
            return "There Is No User Matched The Keywords!"
        return self._get_user_content_by_user_list(matched_users)

    def get_all_post_content(self):
        return self._get_post_content_by_post_list(self.posts.values())

    def get_user_by_user_id(self, user_id: str):
        if user_id in self.users:
            return self.users[user_id]
        else:
            return None

    def get_post_list_content_by_hot_value_page(self, page_id: int, agent: RedditUser):
        post_list_by_hot_value = self.get_sorted_post_list_by_hot_value(
            self.get_posts().values()
        )
        post_list_by_hot_value_page = self._get_post_by_page_id_post_list(
            page_id, post_list_by_hot_value
        )
        if post_list_by_hot_value_page:

            return self._get_post_content_by_post_list(post_list_by_hot_value_page)
        return None

    def get_post_list_number_pagesize_maxpage_by_hot_value(self):
        post_list_by_hot_value = self.get_sorted_post_list_by_hot_value(
            self.get_posts().values()
        )
        count, max_page = self._get_post_list_count_page(post_list_by_hot_value)
        return count, self.page_size, max_page

    def _load_data(self, config):
        self._load_users(config["paths"]["reddit_users"])
        self._load_posts(config["paths"]["reddit_posts"])
        self._load_comments1(config["paths"]["reddit_comments1"])
        self._load_comments2(config["paths"]["reddit_comments2"])
        # self._load_subreddit(config["paths"]["subreddit"])

    def _load_posts(self, file_path: str):
        with open(file_path, newline="", encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                post = RedditPost(
                    row["submission_id"],
                    row["author_id"],
                    row["author_name"],
                    row["posts"],
                    int(row["score"]),
                    int(float(row["num_comments"])),
                    float(row["upvote_ratio"]) if row["upvote_ratio"] else 0.0,
                    row["created_utc"],
                    row["subreddit"],
                )
                self.posts[post.submission_id] = post
                # self.name_submission_id_dict[post.name] = post.submission_id
                # post
                if row["author_id"] in list(self.users.keys()):
                    self.users[row["author_id"]].add_user_post_by_post(post=post)

    def _load_comments1(self, file_path: str):
        with open(file_path, newline="", encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                comment1 = RedditComment1(
                    row["comment_id"],
                    row["comment_author_name"],
                    row["comment_author_id"],
                    # int(row["comment_score"]),
                    row["comment_score"],
                    row["comment_body"],
                    row["link_id"],
                    row["parent_id"],
                    row["subreddit"],
                    row["created_utc"],
                    row["level"]
                )
                self.comments1.append(comment1)
                parent_id = row["parent_id"].split("_")[1]

                # comment posts
                if parent_id in list(self.posts.keys()):
                    self.posts[parent_id].add_comment1(comment1)

                # Comment content is added to the corresponding user's comment
                if row["comment_author_id"] in list(self.users.keys()):
                    self.users[row["comment_author_id"]].add_user_comment_by_commet1(
                        comment=comment1
                    )


    def _load_comments2(self, file_path: str):
        with open(file_path, newline="", encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                comment2 = RedditComment2(
                    row["comment_id"],
                    row["comment_author_name"],
                    row["comment_author_id"],
                    # int(row["comment_score"]),
                    row["comment_score"],
                    row["comment_body"],
                    row["link_id"],
                    row["parent_id"],
                    row["subreddit"],
                    row["created_utc"],
                    row["level"]
                )
                self.comments2.append(comment2)
                parent_id = row["parent_id"].split("_")[1]

                # comment posts
                if parent_id in list(self.posts.keys()):
                    self.posts[parent_id].add_comment2(comment2)

                # Comment content is added to the corresponding user's comment
                if row["comment_author_id"] in list(self.users.keys()):
                    self.users[row["comment_author_id"]].add_user_comment_by_commet2(
                        comment=comment2)

                # 评论内容添加到对应用户的评论中
                if row["comment_author_id"] in list(self.users.keys()):
                    self.users[row["comment_author_id"]].add_user_comment_by_commet2(
                        comment=comment2
                    )


    def _load_users(self, file_path: str):
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                user = RedditUser(
                    row["user_id"],
                    row["name"],
                    row["description"],
                    row["subreddit"],
                    row['submission_num'],
                    row['comment_num'],
                    row['character_setting'],
                    row['comment_num_1'],
                    row['comment_num_2']
                )
                self.users[user.author_id] = user


    def _get_post_by_post_id(self, post_id: str):
        if post_id in self.posts.keys():
            return self.posts[post_id]
        else:
            return None

    def _get_post_content_by_post_id(self, post_id: str):
        if post_id in self.posts:
            res = {
                "submission_id": self.posts[post_id].submission_id,
                "author_name": self.posts[post_id].author_name,
                "posts": self.posts[post_id].posts,
                "selftext": self.posts[post_id].selftext,
                "time": self.posts[post_id].time,
                "subreddit": self.posts[post_id].subreddit,

            }
        else:
            res = f"Post {self.posts[post_id].submission_id} Not Exist!"
        return res

    def _get_post_content_by_post_list(self, post_list: List[RedditPost]):
        post_list_content = [
            self._get_post_content_by_post_id(post.submission_id) for post in post_list
        ]
        return list2str(post_list_content)

    def _get_user_content_by_author_id(self, author_id: str):
        if author_id in self.users:
            res = {
                "author_id": self.users[author_id].author_id,
                "author_name": self.users[author_id].author_name,
                "description": self.users[author_id].description,
                "subreddit": self.users[author_id].subreddit
            }
        else:
            res = ""
        return res

    def _get_user_content_by_user_list(self, user_list: List[RedditUser]):
        user_list_content = [
            self._get_user_content_by_author_id(user.author_id) for user in user_list
        ]
        return list2str(user_list_content)

    def _get_post_list_count_page(self, post_list: List[Any]):
        assert self.page_size != 0
        max_page = len(post_list) // self.page_size + (
            1 if len(post_list) % self.page_size > 0 else 0
        )
        return len(post_list), max_page

    def _get_post_by_page_id_post_list(self, page_id: int, post_list: List[Any]):
        assert self.page_size != 0
        max_page = len(post_list) // self.page_size + (
            1 if len(post_list) % self.page_size > 0 else 0
        )
        if 0 <= page_id <= max_page:
            return post_list[page_id * self.page_size : (page_id + 1) * self.page_size]
        return None

    def _get_post_hot_value_by_post_id(self, post_id: str):
        cur_post = self._get_post_by_post_id(post_id)
        if cur_post:
            post_hot_value = hot_by_score_time(cur_post.score, cur_post.time)
            return post_hot_value
        return None

    def get_post_hot_value_by_post_list(self, post_list: List[RedditPost]):
        hot_value_res = {}
        for post in post_list:
            post_id = post.submission_id
            hot_value_res[post_id] = self._get_post_hot_value_by_post_id(post_id)
        return hot_value_res

    def get_sorted_post_list_by_hot_value(
        self, post_list: List[RedditPost]
    ) -> List[RedditPost]:
        sort_dict = self.get_post_hot_value_by_post_list(post_list)
        post_dict = {post.submission_id: post for post in post_list}
        return self.get_sorted_post_list_by_sort_dict(sort_dict, post_dict)

    def get_sorted_post_list_content_by_hot_value(self, post_list: List[RedditPost]):
        return self._get_post_content_by_post_list(
            self.get_sorted_post_list_by_hot_value(post_list)
        )

    def get_sorted_post_list_by_sort_dict(
        self, sort_dict: Dict[str, Any], sort_post_list: Dict[str, Any]
    ) -> List[RedditPost]:
        sort_id = sorted(sort_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_posts = [sort_post_list[post_id] for post_id, _ in sort_id]
        return sorted_posts



    def add_user(self, user: RedditUser):
        self.users[user.author_id] = user

    def get_posts(self):
        return self.posts

    def get_comments1(self):
        return self.comments1

    def get_comments2(self):
        return self.comments2


