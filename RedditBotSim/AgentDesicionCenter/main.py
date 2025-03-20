import json
import random
from langchain_openai import ChatOpenAI
from reddit_agent import create_reddit_user, RedditAgent
import sys
sys.path.append('./RedditBotSimEnvironment')
from env import RedditEnv
sys.path.append('./RedditBotSimUtils')
from utils import load_config, set_logger
import time_utils as time_utils
from tqdm import tqdm
import pandas as pd
import os
from datetime import datetime, timedelta
import ast 
from openai import OpenAI
import itertools

def main(logger, client, config, env, model, temperature):
    logger.info(f'Program start time:{time_utils.get_now_time()}')
    logger.info(" ")
    begin_time = time_utils.get_now_time()

    # Read the CSV file of the agent user
    # agent_profiles.csv file: We used LLM and real user statistics to generate basic information about the bot account in advance, Including "user_id, user_name, Age, Gender, Region and EducationLevel, Ideology, the Description, Preference, SubRedditNum, SubRedditList, PostNum, Comment1Num Comment2Num, Formulate_Post Formulate_Comment1, Formulate_Comment2". These bot accounts are then created and stored in the overall user pool (Users.csv).
    # Users.csv: Users.csv is a collection of basic user information. The environment runs initially with only human account data, and then adds one piece of data to this file with each new account created.
    total_user_nums = config["agents_num"]    
    df_agent_user = pd.read_csv(config["paths"]["agent_bots"], encoding='utf-8') # agent_profiles.csv
    # Create Reddit Agents
    for user_num in tqdm(range(total_user_nums)) :
        logger.info(f'Start time of {user_num} user:{time_utils.get_now_time()}')
        logger.info(" ")

        # Create Users
        agent_columns = ['user_id', 'user_name', 'Description', 'SubRedditList', 'PostNum', 'Comment1Num', 'Comment2Num', 'Preference']
        new_agent_user = df_agent_user.loc[user_num, agent_columns]
        reddit_user, PostNum, CommentNum1, CommentNum2, user_character_setting, user_subreddit = create_reddit_user(env, new_agent_user)
        reddit_agent = RedditAgent(reddit_user, env)   # An agent bot data will be added to the Users.csv file

        # Profile
        row = df_agent_user.iloc[user_num]
        UserName = row['user_name']
        Gender = row['Gender']
        Age = row['Age']
        Region = row['Region']
        Education = row['EducationLevel']
        Ideology = row['Ideology']
        Description = row['Description']
        Event = row['Preference']


        # PostAction
        # if isinstance(row['Formulate_Post'], list):
        #     Formulate_Post = row['Formulate_Post']
        # else:
        #     Formulate_Post = ast.literal_eval(row['Formulate_Post'])

        # for post_index, post_info in enumerate(Formulate_Post):
        #     # logger.info(f'The Posting start time of user {user_num}: {time_utils.get_now_time()}')
        #     # logger.info("************")
        #     planed_subreddit = post_info[2]
        #     planed_preference = post_info[3]
        #     planed_time = post_info[1]
        #     length = ast.literal_eval(row['PostLengthList'])[post_index]
        #     post_time = reddit_agent.set_posttime(planed_time)
        #     Knowledge = reddit_agent.select_post(planed_preference, post_time, config)
        #     Prompt_post = f"""You are {UserName}, your gender is {Gender}, your age is {Age}, you are from {Region}, your education level is {Education}, your political ideology is {Ideology}, your personal description is {Description}, and you are primarily interested in {planed_preference} events. You need to generate a post about {planed_preference} news. Additional background information on this news is [Knowledge]. You need to summarize the background [Knowledge] and generate your [Response]. Your [Response] should match your personal information and your response must refer to [Knowledge]. [Response] must be a string of characters close to the {length} character.
            
        #         [Knowledge]:
        #             {Knowledge}
                
        #         [Response]:
        #     """
            
        #     new_post = reddit_agent.post(client, model, Prompt_post, temperature, post_time=post_time, subreddit = planed_subreddit)
        #     print(post_index)
        
         # Comment1Action
        if isinstance(row['Formulate_Comment1'], list):
            Formulate_Comment1 = row['Formulate_Comment1']
        else:
            Formulate_Comment1 = ast.literal_eval(row['Formulate_Comment1'])
        for comment1_index, comment1_info in enumerate(Formulate_Comment1):
            logger.info(f'The Comment 1 start time of user {user_num}: {time_utils.get_now_time()}')
            logger.info("************")
            print(comment1_index)
            planed_subreddit = comment1_info[2]
            planed_preference = comment1_info[3]
            planed_time = comment1_info[1]
            browse_posts_time, browse_end_posts_time = reddit_agent.set_read_post_time(planed_time, config)            
            recommend_posts = reddit_agent.read(browse_posts_time, browse_end_posts_time)
            if len(recommend_posts) > 0:
                recommend_posts_id_list = []
                recommend_posts_time_list = []
                for posts_ids in recommend_posts:
                    recommend_posts_id_list.append(posts_ids.submission_id)
                    recommend_posts_time_list.append(posts_ids.time)
                
                ImitateComment =  reddit_agent.comment_imitate(config["paths"]["reddit_comments1"])
                count = 0
                commendposts = reddit_agent.get_recommend_posts(recommend_posts, count)
                # with open(config["paths"]["Prompt_Comment1"], 'r', encoding='utf-8') as file:
                #     content = file.read()
                
                Prompt_Comment1 = f"""You are {UserName}, your gender is {Gender}, your age is {Age}, you are from {Region}, your education level is {Education}, your political ideology is {Ideology}, your personal description is {Description}, and you are primarily interested in {planed_preference} events. 
                You must comment a post about {planed_preference} news. You need to read the content of the [Post], and reply to the post. 
                Your [Response] must generate a comment on [Post] that mimics the linguistic style, length, and structure of the sentences in [Imitation]. 
                You need to provide the "Post ID" and "Post Time" of the Post and the content of your comment. The format of [Response] must be a format such as showed in [Format]. [Rules] have a higher priority.

                Please read 5 posts. Here are the details of each post:
                [Posts]:
                    {commendposts}

                [Imitation]: 
                    {ImitateComment}

                [Rules]
                    1. When [Posts] is empty, [Response] is "End".  
                    2. When [Posts] is not related to {planed_preference} information, [Response] is "Continue browsing".
                    3. Your [Response] must generate a comment on [Post] that mimics the linguistic style, length, and structure of the sentences in [Imitation]. 
                    4. You must comment a post about {planed_preference} news.

                [Format]:
                    ["PostID", "Post Time", "Comment Content"]
                    
                [Response]:"""
                
                new_comment = reddit_agent.comment1_content(client, model, temperature, Prompt_Comment1, recommend_posts_id_list, recommend_posts_time_list)
                i = 1
                while i:
                    if new_comment == "Continue browsing":
                        count = count + 5
                        commendposts = reddit_agent.get_recommend_posts(recommend_posts, count)
                        Prompt_Comment1 = f"""You are {UserName}, your gender is {Gender}, your age is {Age}, you are from {Region}, your education level is {Education}, your political ideology is {Ideology}, your personal description is {Description}, and you are primarily interested in {Event} events. 
                        You must comment a post about {Event} news. You need to read the content of the [Post], and reply to the post. 
                        Your [Response] must generate a comment on [Post] that mimics the linguistic style, length, and structure of the sentences in [Imitation]. 
                        You need to provide the "Post ID" and "Post Time" of the Post and the content of your comment. The format of [Response] must be a format such as showed in [Format]. [Rules] have a higher priority.

                        Please read 5 posts. Here are the details of each post:
                        [Posts]:
                            {commendposts}

                        [Imitation]: 
                            {ImitateComment}

                        [Rules]
                            1. When [Posts] is empty, [Response] is "End".  
                            2. When [Posts] is not related to {Event} information, [Response] is "Continue browsing".
                            3. Your [Response] must generate a comment on [Post] that mimics the linguistic style, length, and structure of the sentences in [Imitation]. 
                            4. You must comment a post about {Event} news.

                        [Format]:
                            ["PostID", "Post Time", "Comment Content"]
                            
                        [Response]:"""
                        new_comment = reddit_agent.comment1_content(client, model, temperature, Prompt_Comment1, recommend_posts_id_list, recommend_posts_time_list)
                    if new_comment == "End" or new_comment == None:
                        new_comment = []
                        selected_post = random.choice(recommend_posts)
                        new_comment.append(selected_post.submission_id)
                        new_comment.append(selected_post.time)
                        new_comment.append(selected_post.posts)
                        break
                    else:
                        break
                
                comment1 = reddit_agent.comment1(new_comment, planed_subreddit)
                if comment1 == None:
                    new_comment = []
                    selected_post = random.choice(recommend_posts)
                    new_comment.append(selected_post.submission_id)
                    new_comment.append(selected_post.time)
                    new_comment.append(selected_post.posts)
                    comment1 = reddit_agent.comment1(new_comment, planed_subreddit)
                
        # Comment2Action
        if isinstance(row['Formulate_Comment2'], list):
            Formulate_Comment2 = row['Formulate_Comment2']
        else:
            Formulate_Comment2 = ast.literal_eval(row['Formulate_Comment2'])
        for comment2_index, comment2_info in enumerate(Formulate_Comment2):
            logger.info(f'The Comment 2 start time of user {user_num}: {time_utils.get_now_time()}')
            logger.info("************")
            planed_subreddit = comment2_info[2]
            planed_preference = comment2_info[3]
            planed_time = comment2_info[1]
            browse_posts_time, browse_end_posts_time = reddit_agent.set_read_post_time(planed_time, config)            
            recommend_posts = reddit_agent.read(browse_posts_time, browse_end_posts_time)            
            recommend_posts_id_list = []
            recommend_comments_id_list = []
            recommend_comments_time_list = []
            recommend_comments = []
            if len(recommend_posts) > 0:
                for posts_ids in recommend_posts:
                    recommend_posts_id_list.append(posts_ids.submission_id)
                    if posts_ids.comments1 != None:
                        for comment_ids in posts_ids.comments1:
                            recommend_comments_id_list.append(comment_ids.comment1_id)
                            recommend_comments_time_list.append(comment_ids.comment1_time)
                            recommend_comments.append(comment_ids)
                
                ImitateComment = reddit_agent.comment_imitate(config["paths"]["reddit_comments1"])
                count = 0
                commendposts = reddit_agent.get_recommend_posts(recommend_posts, count)
                # with open(config["paths"]["Prompt_Comment1"], 'r', encoding='utf-8') as file:
                #     content = file.read()
                
                Prompt_Comment2 = f"""You are {UserName}, your gender is {Gender}, your age is {Age}, you are from {Region}, your education level is {Education}, your political ideology is {Ideology}, your personal description is {Description}, and you are primarily interested in {planed_preference} events. 
                You must comment a First-Level comment about {planed_preference} news. You need to read the content of the [Post], and reply to the First-Level Comment. 
                Your [Response] must generate a comment on [Post] that mimics the linguistic style, length, and structure of the sentences in [Imitation]. 
                You need to provide the "Post ID", "First-Level Comment ID" and "First-Level Comment Time" of your comment as well as the content of your comment on First-Level Comment. The format of [Response] must be the format shown by [format].
                "First-Level Comment Time" must belong to the same First-Level Comment as "First-Level Comment ID".
                [Rules] have a higher priority.

                Please read 5 posts. Here are the details of each post:
                [Posts]:
                    {commendposts}

                [Imitation]: 
                    {ImitateComment}

                [Rules]
                    1. When [Posts] is empty, [Response] is "End".  
                    2. When [Posts] is not related to {planed_preference} information, [Response] is "Continue browsing".
                    3. Your [Response] must generate a comment on [Post]'s First-Level Comment that mimics the linguistic style, length, and structure of the sentences in [Imitation]. 
                    4. You must comment on a First-Level comment about {planed_preference} news.
                    5. The post you select must be about {planed_preference} news.
                    6. First-Level comments must have commented on the corresponding post.
                    7. The [Response] is output in [Format] by filling the corresponding value into [Format]. The output of the [Response] does not need to retain the parameters "Post ID", "First-Level Comment ID", "First-Level Comment Time", "Comment Content".
                    8. "First-Level Comment Time" must belong to the same First-Level Comment as "First-Level Comment ID".
                    9. Your [Response] must generate a comment on [Post] that mimics the linguistic style, length, and structure of the sentences in [Imitation]. 

                [Format]:
                    ["Post ID", "First-Level Comment ID", "First-Level Comment Time", "Comment Content"]
                    
                [Response]:"""
                
                new_comment = reddit_agent.comment2_content(client, model, temperature, Prompt_Comment2, recommend_posts_id_list, recommend_comments_id_list, recommend_comments_time_list)
                i = 1
                while i:
                    if new_comment == "Continue browsing":
                        count = count + 5
                        commendposts = reddit_agent.get_recommend_posts(recommend_posts, count)
                        Prompt_Comment2 = f"""You are {UserName}, your gender is {Gender}, your age is {Age}, you are from {Region}, your education level is {Education}, your political ideology is {Ideology}, your personal description is {Description}, and you are primarily interested in {planed_preference} events. 
                        You must comment a First-Level comment about {planed_preference} news. You need to read the content of the [Post], and reply to the First-Level Comment. 
                        Your [Response] must generate a comment on [Post] that mimics the linguistic style, length, and structure of the sentences in [Imitation]. 
                        You need to provide the "Post ID", "First-Level Comment ID" and "First-Level Comment Time" of your comment as well as the content of your comment on First-Level Comment. The format of [Response] must be the format shown by [format].
                        "First-Level Comment Time" must belong to the same First-Level Comment as "First-Level Comment ID".
                        [Rules] have a higher priority.

                        Please read 5 posts. Here are the details of each post:
                        [Posts]:
                            {commendposts}

                        [Imitation]: 
                            {ImitateComment}

                        [Rules]
                            1. When [Posts] is empty, [Response] is "End".  
                            2. When [Posts] is not related to {planed_preference} information, [Response] is "Continue browsing".
                            3. Your [Response] must generate a comment on [Post]'s First-Level Comment that mimics the linguistic style, length, and structure of the sentences in [Imitation]. 
                            4. You must comment on a First-Level comment about {planed_preference} news.
                            5. The post you select must be about {planed_preference} news.
                            6. First-Level comments must have commented on the corresponding post.
                            7. The [Response] is output in [Format] by filling the corresponding value into [Format]. The output of the [Response] does not need to retain the parameters "Post ID", "First-Level Comment ID", "First-Level Comment Time", "Comment Content".
                            8. "First-Level Comment Time" must belong to the same First-Level Comment as "First-Level Comment ID".
                            9. Your [Response] must generate a comment on [Post] that mimics the linguistic style, length, and structure of the sentences in [Imitation]. 
                            
                        [Format]:
                            ["Post ID", "First-Level Comment ID", "First-Level Comment Time", "Comment Content"]
                            
                        [Response]:"""
                        new_comment = reddit_agent.comment2_content(client, model, temperature, Prompt_Comment2, recommend_posts_id_list, recommend_comments_id_list, recommend_comments_time_list)
                    if new_comment == "End" or new_comment == None:
                        new_comment = []
                        selected_post = random.choice(recommend_posts)
                        new_comment.append(selected_post.submission_id)
                        if selected_post.comments1 != None:
                            selected_comment = random.choice(selected_post.comments1)
                            new_comment.append(selected_comment.comment1_id) 
                            new_comment.append(selected_comment.comment1_time)
                            new_comment.append(selected_comment.comment1_body)
                        break
                    else:
                        break
                if len(new_comment) == 4:
                    comment2 = reddit_agent.comment2(new_comment, planed_subreddit)
                else:
                    break
                i = 1
                while i:
                    if comment2 == None:
                        new_comment = []
                        selected_post = random.choice(recommend_posts)
                        new_comment.append(selected_post.submission_id)
                        if selected_post.comments1 != None:
                            selected_comment = random.choice(selected_post.comments1)
                            new_comment.append(selected_comment.comment1_id) 
                            new_comment.append(selected_comment.comment1_time)
                            new_comment.append(selected_comment.comment1_body)
                            comment2 = reddit_agent.comment2(new_comment, planed_subreddit)
                            break
                        else: 
                            continue
                    else:
                        break        
        
        logger.info(" ")
        logger.info(f'End time of {user_num} user:{time_utils.get_now_time()}')
                # new_post = reddit_agent.post(client, model, Prompt_post, temperature, post_time=post_time, subreddit = planed_subreddit)

    


if __name__ == '__main__':
    # Create Reddit Env
    
    model='gpt-4o-mini'
    temperature=1.0
    openai_api_base=' '
    openai_api_key=' '
    client = OpenAI(
            api_key = openai_api_key,
            base_url = openai_api_base
            )
    config = load_config("./RedditBotSim/config.yml")

    env = RedditEnv(config)
    env.reset()
    logger = set_logger(config)
    main(logger, client, config, env, model, temperature)