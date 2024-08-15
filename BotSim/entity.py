from dataclasses import dataclass
from datetime import datetime
import csv
from typing import List, Dict, Optional


@dataclass
class Post:
    submission_id: str
    author_id: str
    author_name: str
    title: str
    selftext: str
    url: str
    is_original_content: bool
    is_self: bool
    name: str
    score: int
    time: datetime
    subreddit: str
    comments: List["Comment"] = None

    def score_add_one(self):
        self.score += 1

    def score_minus_one(self):
        self.score -= 1

    def add_comment(self, comment: "Comment"):
        if self.comments is None:
            self.comments = []
        self.comments.append(comment)

    def get_num_comments(self) -> int:
        return len(self.comments) if self.comments else 0

    def save_to_file(self, filename: str):
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    None,
                    self.submission_id,
                    self.author_id,
                    self.author_name,
                    self.title,
                    self.selftext,
                    self.url,
                    self.is_original_content,
                    self.is_self,
                    self.name,
                    self.score,
                    self.get_num_comments(),
                    None,
                    self.time,
                    self.subreddit,
                ]
            )


@dataclass
class Comment:
    author_id: str
    score: int
    body: str
    time: datetime
    parent_id: str
    author_name: str

    def save_to_file(self, filename: str):
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    None,
                    self.author_name,
                    self.author_id,
                    self.score,
                    self.body,
                    None,
                    None,
                    self.parent_id,
                    None,
                    None,
                    self.time,
                ]
            )


@dataclass
class User:
    author_id: str
    author_name: str
    created_utc: int
    icon_img: str
    user_posts: Dict[str, Post] = None
    user_comments: List[Comment] = None
    read_posts: List[Post] = None
    follow_users: List["User"] = None

    def __post_init__(self):
        if self.user_posts is None:
            self.user_posts = {}
        if self.user_comments is None:
            self.user_comments = []
        if self.read_posts is None:
            self.read_posts = []
        if self.follow_users is None:
            self.follow_users = []

    def mark_post_as_read(self, post: Post):
        self.read_posts.append(post)

    def add_user_post_by_post(self, post: Post):
        self.user_posts[post.submission_id] = post

    def add_user_comment_by_commet(self, comment: Comment):
        self.user_comments.append(comment)

    def add_user_have_read_post_by_post(self, post: Post):
        self.read_posts.append(post)

    def add_user_follow_by_user(self, user: "User"):
        self.follow_users.append(user)

    def save_have_read_post_to_file(self, filename: str, post: Post):
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.author_id, post.submission_id])

    def save_follow_user_to_file(self, filename: str, user: "User"):
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.author_id, user.author_id])

    def save_to_file(self, filename: str):
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    None,
                    self.author_id,
                    None,
                    self.created_utc,
                    self.icon_img,
                    None,
                    None,
                    None,
                    None,
                    self.author_name,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ]
            )
