from dataclasses import dataclass
from datetime import datetime
import csv
from typing import List, Dict, Optional

@dataclass
class SubReddit:
    SubReddit_id: str
    SubReddit_name: str
    SubReddit_description: str
    subscribers: int
    created_utc: str


@dataclass
class RedditPost:
    submission_id: str
    author_id: str
    author_name: str
    posts: str
    score: int
    num_comments: int
    upvote_ratio:float
    time: datetime
    subreddit: str
    comments1: List["RedditComment1"] = None
    comments2: List["RedditComment2"] = None

    def score_add_one(self):
        self.score += 1

    def score_minus_one(self):
        self.score -= 1

    def add_upvote_ratio(self):
        self.upvote_ratio = self.score / ((self.score - 1) / self.upvote_ratio)

    def minus_upvote_ratio(self):
        self.upvote_ratio = self.score / ((self.score + 1) / self.upvote_ratio)

    def add_comment1(self, comment: "RedditComment1"):
        if self.comments1 is None:
            self.comments1 = []
        self.comments1.append(comment)

    def add_comment2(self, comment: "RedditComment2"):
        if self.comments2 is None:
            self.comments2 = []
        self.comments2.append(comment)

    def get_num_comments(self) -> int:
        return len(self.comments) if self.comments else 0

    def save_to_file(self, filename: str):
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.submission_id,
                    self.author_id,
                    self.author_name,
                    self.posts,
                    self.score,
                    self.num_comments,
                    self.upvote_ratio,
                    self.time,
                    self.subreddit,
                ]
            )


@dataclass
class RedditComment1:
    comment1_id: str
    comment1_author_name: str
    comment1_author_id: str
    comment1_score: int
    comment1_body: str
    link_id: str
    parent_id: str
    subreddit: str
    comment1_time: str
    level: str

    def save_to_file(self, filename: str):
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.comment1_id,
                    self.comment1_author_id,
                    self.comment1_author_name,                    
                    self.comment1_score,
                    self.comment1_body,                    
                    self.link_id,
                    self.parent_id,
                    self.subreddit,
                    self.comment1_time,
                    self.level
                ]
            )

@dataclass
class RedditComment2:
    comment2_id: str
    comment2_author_name: str
    comment2_author_id: str
    comment2_score: int
    comment2_body: str
    link_id: str
    parent_id: str
    subreddit: str
    comment2_time: str
    level: str

    def save_to_file(self, filename: str):
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.comment2_id,
                    self.comment2_author_id,
                    self.comment2_author_name,                    
                    self.comment2_score,
                    self.comment2_body,                    
                    self.link_id,
                    self.parent_id,
                    self.subreddit,
                    self.comment2_time,
                    self.level
                ]
            )

@dataclass
class RedditUser:
    author_id: str
    author_name: str
    description: str
    subreddit: str
    submission_num: str
    comment_num: str
    character_setting: str
    comment_num_1: str
    comment_num_2: str
    follow_subReddit: Dict[str,SubReddit] = None
    user_posts: Dict[str, RedditPost] = None
    user_comments1: List[RedditComment1] = None
    user_comments2: List[RedditComment2] = None
    

    def __post_init__(self):
        if self.user_posts is None:
            self.user_posts = {}
        if self.user_comments1 is None:
            self.user_comments1 = []
        if self.user_comments2 is None:
            self.user_comments2 = []

        if self.follow_subReddit is None:
            self.follow_subReddit = {}

    def add_user_post_by_post(self, post: RedditPost):
        self.user_posts[post.submission_id] = post

    def add_user_comment_by_commet1(self, comment: RedditComment1):
        self.user_comments1.append(comment)

    def add_user_comment_by_commet2(self, comment: RedditComment2):
        self.user_comments2.append(comment)

    def add_follow_subreddit(self, subReddit: SubReddit):
        self.follow_subReddit[subReddit.SubReddit_name] = subReddit

    def save_follow_subreddit_to_file(self, filename: str, follow_subreddit: str):
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.author_id, follow_subreddit])

    def save_to_file(self, filename: str):
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.author_id,
                    self.author_name,
                    self.description,
                    self.submission_num,
                    self.comment_num,
                    self.character_setting,
                    self.comment_num_1,
                    self.comment_num_2,
                    self.subreddit
                ]
            )

