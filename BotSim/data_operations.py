import csv
from entity import Post, Comment, User
from datetime import datetime
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


class RedditData:
    """
    Data class.
    """

    def __init__(self, config):
        self.config = config
        self.posts: Dict[str, Post] = {}
        self.comments: List[Comment] = []
        self.users: Dict[str, User] = {}
        self.name_submission_id_dict: Dict[str, str] = {}
        self.page_size: int = self.config["constants"]["page_size"]
        self.subreddit_list: List[str] = []
        self._load_data(config)

    def add_post_by_subreddit_title_content_agent(
        self,
        subreddit: str,
        title: str,
        content_text: str,
        content_url: str,
        is_original_content: bool,
        is_self: bool,
        time: str,
        agent: User,
    ):
        time = str_to_datetime(time_str=time)
        if not check_time(cur_time=time, config=self.config):
            start_time = load_start_time(config=self.config)
            end_time = load_end_time(config=self.config)
            return f"Input time is {time}. Start time is {start_time}, end time is {end_time}. The User must conduct their actions within the specified time frame."
        if subreddit not in self.subreddit_list:
            return "Subreddit Is Wrong!"
        existing_post_id = list(self.get_posts().keys())
        submission_id = generate_reddit_id(existing_post_id)
        add_post = Post(
            submission_id=submission_id,
            author_id=agent.author_id,
            author_name=agent.get_author_name(),
            title=title,
            selftext=content_text,
            url=content_url,
            is_original_content=is_original_content,
            is_self=is_self,
            name=agent.author_id,
            score=0,
            time=time,
            subreddit=subreddit,
        )
        self.posts[submission_id] = add_post
        agent.add_user_post_by_post(add_post)
        agent.mark_post_as_read(add_post)
        add_post.save_to_file(filename=self.config["paths"]["posts"])
        agent.save_have_read_post_to_file(
            filename=self.config["paths"]["have_read_posts"], post=add_post
        )
        return f"Post Success! The Post ID is {submission_id}"

    def add_user_by_user_name_description(
        self, user_name: str, icon_img: str, time: str
    ):
        time = str_to_datetime(time_str=time)
        if not check_time(cur_time=time, config=self.config):
            start_time = load_start_time(config=self.config)
            end_time = load_end_time(config=self.config)
            return f"Input time is {time}. Start time is {start_time}, end time is {end_time}. The User must conduct their actions within the specified time frame."
        existing_user_id = list(self.users.keys())
        add_user_id = generate_reddit_id(existing_ids=existing_user_id, length=5)
        created_utc = int(time.timestamp())
        add_user = User(
            author_id=add_user_id,
            author_name=user_name,
            created_utc=created_utc,
            icon_img=icon_img,
        )
        self.users[add_user_id] = add_user
        add_user.save_to_file(filename=self.config["paths"]["users"])
        return (f"Create User Sucess! Your User ID is {add_user_id}", add_user)

    def add_comment_by_parent_id_content_agent(
        self, parent_id: str, content: str, time: str, agent: User
    ):
        time = str_to_datetime(time_str=time)
        if not check_time(cur_time=time, config=self.config):
            start_time = load_start_time(config=self.config)
            end_time = load_end_time(config=self.config)
            return f"Input time is {time}. Start time is {start_time}, end time is {end_time}. The User must conduct their actions within the specified time frame."
        submission_id = parent_id
        if agent == None:
            return "User Not Exist!"
        if submission_id in self.posts.keys():
            add_exe_comment = Comment(
                author_name=agent.author_name,
                author_id=agent.author_id,
                score=0,
                body=content,
                time=time,
                parent_id=self.posts[submission_id].name,
            )
            self.posts[submission_id].add_comment(add_exe_comment)
            self.comments.append(add_exe_comment)
            agent.add_user_comment_by_commet(add_exe_comment)
            add_exe_comment.save_to_file(
                filename=self.config["paths"]["comments"]
            )
            return "Comment Success!"
        return "Comment Post Not Exist!"

    def follow_user_by_user_id(self, user_id: str, agent: User):
        if user_id == agent.author_id:
            return "Can't Follow Self!"
        if user_id in self.users.keys():
            agent.add_user_follow_by_user(self.users[user_id])
            agent.save_follow_user_to_file(
                filename=self.config["paths"]["user_follow"],
                user=self.users[user_id],
            )
            return f"Follow {user_id} Success!"
        else:
            return "Follow User Not Exists!"

    def like_post_by_post_id(self, post_id: str):
        if post_id in self.posts.keys():
            self.posts[post_id].score_add_one()
            return "Like Post Success!"
        return "Like Post Not Exist!"

    def dislike_post_by_post_id(self, post_id: str):
        if post_id in self.posts.keys():
            self.posts[post_id].score_minus_one()
            return "Dislike Post Success!"
        return "Dislike Post Not Exist!"

    def search_post_by_keywords(self, keywords: str):
        matched_posts = []
        keywords_list = keywords.split()
        for post in self.posts.values():
            for keyword in keywords_list:
                if keyword in post.title or keyword in post.selftext:
                    matched_posts.append(post)
                    break
        if len(matched_posts) == 0:
            return "There Is No Post Matched The Keywords!"
        return self._get_post_content_by_post_list(matched_posts)

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

    def get_post_list_content_by_hot_value_page(self, page_id: int, agent: User):
        post_list_by_hot_value = self.get_sorted_post_list_by_hot_value(
            self.get_posts().values()
        )
        post_list_by_hot_value_page = self._get_post_by_page_id_post_list(
            page_id, post_list_by_hot_value
        )
        if post_list_by_hot_value_page:
            if agent:
                for post in post_list_by_hot_value_page:
                    agent.mark_post_as_read(post)
                    agent.save_have_read_post_to_file(
                        filename=self.config["paths"]["have_read_posts"], post=post
                    )
            return self._get_post_content_by_post_list(post_list_by_hot_value_page)
        return None

    def get_post_list_number_pagesize_maxpage_by_hot_value(self):
        post_list_by_hot_value = self.get_sorted_post_list_by_hot_value(
            self.get_posts().values()
        )
        count, max_page = self._get_post_list_count_page(post_list_by_hot_value)
        return count, self.page_size, max_page

    def _load_data(self, config):
        self._load_users(config["paths"]["users"])
        self._load_posts(config["paths"]["posts"])
        self._load_comments(config["paths"]["comments"])
        self._load_have_read(config["paths"]["have_read_posts"])
        self._load_follow_relation(config["paths"]["user_follow"])
        self._load_subreddit()

    def _load_subreddit(self):
        self.subreddit_list = self.config["subreddit"]

    def _load_posts(self, file_path: str):
        with open(file_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                post = Post(
                    row["submission_id"],
                    row["author_id"],
                    row["author_name"],
                    row["title"],
                    row["selftext"],
                    row["url"],
                    bool(row["is_original_content"]),
                    bool(row["is_self"]),
                    row["name"],
                    int(row["score"]),
                    row["time"],
                    row["subreddit"],
                )
                self.posts[post.submission_id] = post
                self.name_submission_id_dict[post.name] = post.submission_id
                if row["author_id"] in self.users.keys():
                    self.users[row["author_id"]].add_user_post_by_post(post=post)

    def _load_comments(self, file_path: str):
        with open(file_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                comment = Comment(
                    row["comment_author_name"],
                    row["comment_author_id"],
                    int(row["comment_score"]),
                    row["comment_body"],
                    row["comment_time"],
                    row["parent_id"],
                )
                self.comments.append(comment)
                # print(row["parent_id"])
                # print(self.name_submission_id_dict)
                if row["parent_id"] in self.name_submission_id_dict.keys():
                    submission_id = self.name_submission_id_dict[row["parent_id"]]
                    # print(submission_id)
                    if submission_id in self.posts.keys():
                        self.posts[submission_id].add_comment(comment)
                    if submission_id in self.users.keys():
                        self.users[row["author_id"]].add_user_comment_by_commet(
                            comment=comment
                        )

    def _load_users(self, file_path: str):
        with open(file_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                user = User(
                    row["user_id"], row["name"], row["created_utc"], row["icon_img"]
                )
                self.users[user.author_id] = user

    def _load_have_read(self, file_path: str):
        with open(file_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                user_id = row["author_id"]
                submission_id = row["submission_id"]
                posts = self._get_post_by_post_id(post_id=submission_id)
                self.users[user_id].add_user_have_read_post_by_post(posts)

    def _load_follow_relation(self, file_path: str):
        with open(file_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                user_id = row["author_id"]
                follow_author_id = row["follow_author_id"]
                self.users[user_id].add_user_follow_by_user(
                    self.users[follow_author_id]
                )

    def add_post(self, post: Post):
        self.posts[post.submission_id] = post

    def remove_post_by_post_id(self, post_id: str):
        if post_id in self.posts.keys():
            del self.posts[post_id]

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
                "title": self.posts[post_id].title,
                "selftext": self.posts[post_id].selftext,
                "time": self.posts[post_id].time,
                "subreddit": self.posts[post_id].subreddit,
            }
        else:
            res = f"Post {self.posts[post_id].submission_id} Not Exist!"
        return res

    def _get_post_content_by_post_list(self, post_list: List[Post]):
        post_list_content = [
            self._get_post_content_by_post_id(post.submission_id) for post in post_list
        ]
        return list2str(post_list_content)

    def _get_user_content_by_author_id(self, author_id: str):
        if author_id in self.users:
            res = {
                "author_id": self.users[author_id].author_id,
                "author_name": self.users[author_id].author_name,
                "description": self.users[author_id].icon_img,
            }
        else:
            res = ""
        return res

    def _get_user_content_by_user_list(self, user_list: List[User]):
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

    def get_post_hot_value_by_post_list(self, post_list: List[Post]):
        hot_value_res = {}
        for post in post_list:
            post_id = post.submission_id
            hot_value_res[post_id] = self._get_post_hot_value_by_post_id(post_id)
        return hot_value_res

    def get_sorted_post_list_by_hot_value(
        self, post_list: List[Post]
    ) -> List[Post]:
        sort_dict = self.get_post_hot_value_by_post_list(post_list)
        post_dict = {post.submission_id: post for post in post_list}
        return self.get_sorted_post_list_by_sort_dict(sort_dict, post_dict)

    def get_sorted_post_list_content_by_hot_value(self, post_list: List[Post]):
        return self._get_post_content_by_post_list(
            self.get_sorted_post_list_by_hot_value(post_list)
        )

    def get_sorted_post_list_by_sort_dict(
        self, sort_dict: Dict[str, Any], sort_post_list: Dict[str, Any]
    ) -> List[Post]:
        sort_id = sorted(sort_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_posts = [sort_post_list[post_id] for post_id, _ in sort_id]
        return sorted_posts

    # def get_recommend_post(self):
    #     return

    # def add_comment(self, comment: Comment):
    #     if comment.get_parent_id() in self.posts:
    #         self.posts[comment.get_parent_id()].add_comment(comment)

    def add_user(self, user: User):
        self.users[user.author_id] = user

    def get_posts(self):
        return self.posts


# config = load_config("./config.yaml")
# data = RedditData(config)
# print(data.get_post_list_content_by_hot_value_page(0))
