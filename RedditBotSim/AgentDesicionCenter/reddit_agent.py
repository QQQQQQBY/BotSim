import os
import sys
sys.path.append('./RedditBotSim/Action')
from Action import CreateUserAction, BrowsePosts, PostAction, CommentAction1, BrowseComments, StopAction, CommentAction2
from Basics import Action
sys.path.append('./RedditBotSim/Environment')
from env import RedditEnv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
import random
from Entity import RedditUser, RedditPost
from datetime import datetime, timedelta
import pandas as pd
sys.path.append('./RedditBotSim/Utils')
import time_utils as time_utils
import ast 
from modify_content import modify_content
import re
import calendar



# Create Agent
def create_reddit_user(reddit_env, new_agent_user):
    create_user_action = CreateUserAction()
    action_args = {
            "user_id": new_agent_user['user_id'],
            "name": new_agent_user['user_name'],
            "description": new_agent_user['Description'],
            "subreddit": new_agent_user['SubRedditList'],
            "submission_num": new_agent_user['PostNum'],
            "comment_num": int(new_agent_user['Comment1Num'] + new_agent_user['Comment2Num']),           
            "character_setting": new_agent_user['Preference'],
            "comment_num_1": new_agent_user['Comment1Num'],
            "comment_num_2": new_agent_user['Comment2Num']
        }
    agent_user = reddit_env.step(
            create_user_action, action_args, None
        )

    reddit_user = agent_user[1] 

    # Post Number
    PostNum = int(reddit_user.submission_num)
    # Comment Number
    CommentNum1 = int(reddit_user.comment_num_1)
    CommentNum2 = int(reddit_user.comment_num_2)
    # Perference
    user_character_setting = reddit_user.character_setting
    # SubReddit
    user_subreddit = reddit_user.subreddit
    user_subreddit = ast.literal_eval(user_subreddit)

    return reddit_user, PostNum, CommentNum1, CommentNum2, user_character_setting, user_subreddit


def generate(client, model, prompt, temperature):
    completion = client.chat.completions.create(
                        model = model,
                        messages = [{"role": "user", "content": prompt}],
                        temperature= temperature
                    )
    llm_return_dict = completion.choices[0].message.content
    try:
        return llm_return_dict

    except Exception as e:
        print("======Exception====== /n", e)
        return None

def generate_comment1(client, model, prompt, temperature, recommend_posts_id_list, recommend_posts_time_list):
    completion = client.chat.completions.create(
                        model = model,
                        messages = [{"role": "user", "content": prompt}],
                        temperature= temperature
                    )
    llm_return_dict = completion.choices[0].message.content
    try:
        if llm_return_dict == "Continue browsing" or llm_return_dict == "End":
            return llm_return_dict
        else:
            date_format="%Y-%m-%d %H:%M:%S"            
            matches = re.findall(r'\[.*?\]', llm_return_dict)
            if len(matches) != 0:
                lists = [ast.literal_eval(match) for match in matches][0]
                datetime.strptime(lists[1], date_format)
                if isinstance(lists, list) and len(lists) == 3 and lists[0] in recommend_posts_id_list and lists[1] in recommend_posts_time_list: 
                    return lists
                else:
                    return None
    except Exception as e:
        print("======Exception====== /n", e)
        return None

def generate_comment2(client, model, prompt, temperature, recommend_posts_id_list, recommend_comments_id_list, recommend_comments_time_list):
    completion = client.chat.completions.create(
                        model = model,
                        messages = [{"role": "user", "content": prompt}],
                        temperature= temperature
                    )
    llm_return_dict = completion.choices[0].message.content
    try:
        if llm_return_dict == "Continue browsing" or llm_return_dict == "End":
            return llm_return_dict
        else:
            date_format="%Y-%m-%d %H:%M:%S"            
            matches = re.findall(r'\[.*?\]', llm_return_dict)
            if len(matches) != 0:
                lists = [ast.literal_eval(match) for match in matches][0]
                datetime.strptime(lists[2], date_format)
                if isinstance(lists, list) and len(lists) == 4 and lists[0] in recommend_posts_id_list and lists[1] in recommend_comments_id_list and lists[2] in recommend_comments_time_list: 
                    return lists
                else:
                    return None
    except Exception as e:
        print("======Exception====== /n", e)
        return None

class RedditAgent:
    def __init__(
            self,
            user: RedditUser,
            env: RedditEnv,
    ) -> None:
        self.env = env
        self.user = user

    # Get the posts in order
    def get_recommend_posts(self, recommend_posts, count):
        # Get the posts in order
        commendposts = ""
        for index, readpost in enumerate(recommend_posts[count: count + 5]):
            if index < 5:
                rposts = f"""Post{index}:
                PostID: {readpost.submission_id}. 
                Post Content:{readpost.posts}. 
                Post Time:{readpost.time}. 
                Post User Name: {readpost.author_name}. """
                commendposts = commendposts + rposts
                if readpost.comments1 != None:
                    commendposts = commendposts + f" The number of First-Level Comments totaled {len(readpost.comments1)}:"
                    for c1index, readcomment in enumerate(readpost.comments1):
                        c1posts = f""" First-Level Comment {c1index}: 
                            First-Level Comment ID: {readcomment.comment1_id}
                            First-Level Comment Content: {readcomment.comment1_body}
                            First-Level Comment Time: {readcomment.comment1_time}
                            First-Level Comment User Name: {readcomment.comment1_author_name}"""
                        commendposts = commendposts + c1posts
                        if readpost.comments2 != None:
                            commendposts = commendposts + f"The number of Second-Level Comment Number totaled {len(readpost.comments2)}:"
                            for c2index,readcomment2 in enumerate(readpost.comments2):
                                if readcomment2.parent_id.split("_")[1] == readcomment.comment1_id:
                                    c2posts = f"""Second-Level Comment {c2index}:
                                                    Second-Level Comment ID: {readcomment2.comment2_id}.
                                                    Second-Level Comment Content: {readcomment2.comment2_body}
                                                    Second-Level Comment Time: {readcomment2.comment2_time}.
                                                    Second-Level Comment User Name: {readcomment2.comment2_author_name}"""
                                    commendposts = commendposts + c2posts
    
        return commendposts


    def comment1_content(self, client, model, temperature, comment_prompt, recommend_posts_id_list, recommend_posts_time_list):
        i = 1
        count = 0
        while i:
            count = count + 1
            lists = generate_comment1(client, model, comment_prompt, temperature, recommend_posts_id_list, recommend_posts_time_list)
            if count > 4:
                return None
            if lists is None:
                continue  
            if isinstance(lists, list) or lists == "Continue browsing" or lists == "End":
                i = 0
            else:   
                continue
        return lists  
    
    def comment2_content(self, client, model, temperature, comment_prompt, recommend_posts_id_list, recommend_comments_id_list, recommend_comments_time_list):
        i = 1
        count = 0
        while i:
            count = count + 1
            lists = generate_comment2(client, model, comment_prompt, temperature, recommend_posts_id_list, recommend_comments_id_list, recommend_comments_time_list)
            if count > 7:
                return None
            if lists is None:
                continue  
            if isinstance(lists, list) or lists == "Continue browsing" or lists == "End":
                i = 0
            else:
                continue                 
        return lists  


    def comment1(self, lists, planed_subreddit):
        postid, posttime, commentcontent = lists[0], lists[1], lists[2] 
        try:
            date_format="%Y-%m-%d %H:%M:%S"
            datetime.strptime(posttime, date_format)
            comment_time = self.comment_time_function(posttime)
            postid = "t3_" + postid
            action = CommentAction1()
            action_args = {"link_id": postid,
                        "parent_id": postid,
                        "comment_content": commentcontent,
                        "time": comment_time,
                        "level": 1,
                        "subreddit": planed_subreddit}
            response = self.env.step(action, action_args, self.user)
            return response
        except Exception as e:
            print("======Exception====== /n", e)
            return None
        
    def comment2(self, lists, planed_subreddit):
        postid, commentid, commenttime, commentcontent = lists[0], lists[1], lists[2], lists[3] 
        try:
            date_format="%Y-%m-%d %H:%M:%S"
            datetime.strptime(commenttime, date_format)
            comment_time = self.comment_time_function(commenttime)
            postid = "t3_" + postid
            parent_id = "t1_" + commentid
            action = CommentAction2()
            action_args = {"link_id": postid,
                        "parent_id": parent_id,
                        "comment_content": commentcontent,
                        "time": comment_time,
                        "level": 2,
                        "subreddit": planed_subreddit}
            response = self.env.step(action, action_args, self.user)
            return response
        except Exception as e:
            print("======Exception====== /n", e)
            return None

    def post(self, client, model, Prompt_post, temperature, post_time, subreddit):

        action = PostAction()
        post = generate(client, model, Prompt_post, temperature)
        # response = json.loads(response)
        action_args = {"posts": post, "time": post_time, "subreddit": subreddit}
        observation = self.env.step(action, action_args, self.user)
        return observation

    def read(self, start_time, end_time):
        action = BrowsePosts()
        action_args = {"read_start_time": start_time, "read_end_time": end_time}
        try:
            posts = self.env.step(action, action_args, self.user)        
            posts = list(posts.values())
        except Exception as e:
            print("======Exception====== /n", e) 
            return []
        return posts

    def read_comments(self, start_time, end_time):
        action = BrowseComments()
        action_args = {"read_start_time": start_time, "read_end_time": end_time}
        comments = self.env.step(action, action_args, self.user)     
        return comments

    def set_posttime(self, post_time):    
        random.seed()
        # Calculate the dispatch time according to the weight (calculation: hour minute second)
        time_slots = [
            (0, 3), (3, 6), (6, 9), (9, 12), 
            (12, 15), (15, 18), (18, 21), (21, 23)
        ]
        weights = [7269, 6183, 4297, 3119, 2442, 1797, 3794, 6894]
        chosen_time_slot = random.choices(time_slots, weights=weights, k=1)[0]
        start_hour, end_hour = chosen_time_slot
        h_random = random.randint(start_hour, end_hour - 1)
        m_random = random.randint(0, 59)
        s_random = random.randint(0, 59)        
        # Combine the hour, minute, and second arrays into a time string in the format HH:MM:SS
        hms_string = f"{h_random:02}:{m_random:02}:{s_random:02}"
        post_time = f"{post_time} {hms_string}"
        return post_time


    # Background Knowledge
    def select_post(self, user_character, planed_time, config):
        # select time interval
        select_post_time_end = datetime.strptime(planed_time, '%Y-%m-%d %H:%M:%S').date()
        # before 10 days
        select_post_time_begin = select_post_time_end - timedelta(days=10)

        select_post_time_begin = select_post_time_begin.strftime('%Y-%m-%d')
        select_post_time_end = select_post_time_end.strftime('%Y-%m-%d')

        # select knowledge
        if user_character == "Russia-Ukraine war":
            df = pd.read_csv(config['paths']["Russia_Ukraine"])
        elif user_character == "Israeli-Palestinian conflict":
            df = pd.read_csv(config['paths']["Palestinian_Israeli_conflict"])
        elif user_character == "US politics":
            df = pd.read_csv(config['paths']["US_Politics"])
        else:
            df = pd.read_csv(config['paths']["World_news"])

        # sieve
        df['date'] = pd.to_datetime(df['date'])
        time1 = pd.to_datetime(select_post_time_end)
        time2 = pd.to_datetime(select_post_time_begin)
        filtered_df = df[(df['date'] < time1) & (df['date'] > time2)]
        filtered_df = filtered_df[filtered_df['title'].str.len() < 110]
        if len(filtered_df) < 5:
            post_knowledge = filtered_df['claim'].tolist()
        else:
            post_knowledge = filtered_df.sample(n=5, replace=False)['claim'].tolist()
        
        knowledge = ""
        for i, k in enumerate(post_knowledge):
            knowledge = knowledge + str(i + 1) + ". " + str(k)
        if knowledge == "":
            knowledge = df.sample(n=1, replace=False)['claim'].tolist()
        # rewrite
        post_knowledge = modify_content(knowledge)
        return post_knowledge


    def set_read_post_time(self, start_time, config):
        year, month, day = start_time.split('-')
        year, month, day = int(year), int(month), int(day)
        last_day_of_month = calendar.monthrange(int(year), int(month))[1]
        if day <= calendar.monthrange(year, month)[1]:
            start_time = datetime(year, month, day)
            start_time = start_time.strftime('%Y-%m-%d')
        if day > last_day_of_month:
            day = last_day_of_month    
            start_time = datetime(year, month, day)
            start_time = start_time.strftime('%Y-%m-%d')
        start_time = start_time + " 00:00:00"
        start_time, end_time = time_utils.gap_times(start_time, 1,0,0)
        end_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        start_time_submissions = end_time - timedelta(days=config['constants']['recommendpostsday'])
        start_time_submissions = start_time_submissions.strftime('%Y-%m-%d %H:%M:%S')
        end_time_submissions = end_time.strftime('%Y-%m-%d %H:%M:%S')
        return start_time_submissions, end_time_submissions

    # Select a post that matches the person to comment on
    def select_comment_posts(self, posts_list, user_character_setting):
        # "Russia-Ukraine", "Palestinian-Israeli conflict", "US Politics", "Internation News"
        
        character = random.choice(user_character_setting)
        if character == "Russia_Ukraine":
            keywords = ["Russia", "Ukraine", "Russian", "Putin", "critics"]
        elif character == "Palestinian-Israeli_conflict":
            keywords = ["Gaza", "Israeli", "U.S.", "Hamas", "Palestinian"]
        elif character == "US_Politics":
            keywords = ["Trump", "Biden", "U.S.", "Republician", "Haley"]
        elif character == "World_news":
            keywords = ["Russia", "Ukraine", "Russian", "Putin", "critics", "Gaza", "Israeli", "U.S.", "Hamas", "Palestinian", "Trump", "Biden", "U.S.", "Republician", "Haley", "climate", "weather", "global", "change"]

        matching_posts_indices = [index for index, post in enumerate(posts_list) if any(keyword in post for keyword in keywords)]

        if len(matching_posts_indices) > 0:
            post_index = random.choice(matching_posts_indices)
            post_content = posts_list[post_index]
            return post_index, post_content
        else:
            post_index = random.choices(range(0,len(posts_list)))[0]
            post_content =  posts_list[post_index]
            return post_index, post_content
        
    def comment_time_function(self, target_post_time):
        # Define the time interval and its weight
        time_intervals = {
            "less_than_1_min": (0, 60),  # 0s to 60s
            "1_to_5_min": (60, 5*60),  # 60s to 300s
            "5_to_10_min": (5*60, 10*60),  # 300s to 600s
            "10_to_30_min": (10*60, 30*60),  # 600s to 1800s
            "30_min_to_1_hour": (30*60, 60*60),  # 1800s to 3600s
            "1_to_3_hours": (60*60, 3*60*60),  # 3600s to 10800s
            "3_to_10_hours": (3*60*60, 10*60*60),  # 10800s to 36000s
            "10_to_24_hours": (10*60*60, 24*60*60)  # 36000s to 86400s
        }
        weights = {
            "less_than_1_min": 3009,
            "1_to_5_min": 5060,
            "5_to_10_min": 2769,
            "10_to_30_min": 5727,
            "30_min_to_1_hour": 5727,
            "1_to_3_hours": 8472,
            "3_to_10_hours": 9304,
            "10_to_24_hours": 4818
        }
        selected_interval = random.choices(list(time_intervals.keys()), weights=list(weights.values()), k=1)[0]
        interval_range = time_intervals[selected_interval]
        random_seconds = random.randint(interval_range[0], interval_range[1])
        random_timedelta = timedelta(seconds=random_seconds)
        target_post_time = datetime.strptime(target_post_time, '%Y-%m-%d %H:%M:%S')
        comment_time = target_post_time + random_timedelta
        comment_time = comment_time.strftime('%Y-%m-%d %H:%M:%S')
        return comment_time

    # Select a comment to imitate
    def comment_imitate(self, path):
        # Select a comment and make a mock reply
        comment_list = pd.read_csv(path)['comment_body'].tolist()
        comment_imitation = random.choice(comment_list)
        return comment_imitation
    
    def set_read_comment_time(self, c2):
        c2 = c2 + " 00:00:00"
        start_time, end_time = time_utils.gap_times(c2,1,0,0)
        # Read only seven days of content
        # datetime 
        end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        # Count the date seven days ago
        start_time_comments = end_time - timedelta(days=1)
        start_time_comments = start_time_comments.strftime('%Y-%m-%d %H:%M:%S')
        end_time_comments = end_time.strftime('%Y-%m-%d %H:%M:%S')
        return start_time_comments, end_time_comments