# Select SubReddit of interest based on SubReddit introduction and browsing SubReddit content.

# worldnews'', ``politics'', ``news'', ``InternationalNews'', ``UpliftingNews'' and ``GlobalTalk''

import random
import json
import random
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import pandas as pd
import os
from openai import OpenAI
from datetime import datetime
import random
import string
import ast
import re
import itertools

def generate(client, model, prompt, temperature):
    completion = client.chat.completions.create(
                        model = model,
                        messages = [{"role": "user", "content": prompt}],
                        temperature= temperature
                    )
    llm_return_dict = completion.choices[0].message.content
    try:
        matches = re.findall(r'\[.*?\]', llm_return_dict)
        if len(matches) != 0:
            lists = [ast.literal_eval(match) for match in matches][0]
            if isinstance(lists, list): 
                return lists
            else:
                return None
        else: 
            return None
    except Exception as e:
        print("======Exception====== /n", e)
        return None

def main():
    model='gpt-4o'
    temperature=0.9
    openai_api_base=''
    openai_api_key=''

    client = OpenAI(
            api_key = openai_api_key,
            base_url = openai_api_base
            )

    Agent_csv_file_path = "./Data/LLMAgentProfile/agent_profiles.csv"
    Agent_df = pd.read_csv(Agent_csv_file_path)

    SubReddit_csv_file_path = "./Data/RedditData/SubReddits.csv"
    SubReddit_df = pd.read_csv(SubReddit_csv_file_path)

    sub_reddit_data = SubReddit_df.head(6)
    SubRedditName1 = sub_reddit_data.iloc[0]['SubReddit']
    SubRedditDes1 = sub_reddit_data.iloc[0]['Description']
    SubRedditContent1 = sub_reddit_data.iloc[0]['Content']

    SubRedditName2 = sub_reddit_data.iloc[1]['SubReddit']
    SubRedditDes2 = sub_reddit_data.iloc[1]['Description']
    SubRedditContent2 = sub_reddit_data.iloc[1]['Content']

    SubRedditName3 = sub_reddit_data.iloc[2]['SubReddit']
    SubRedditDes3 = sub_reddit_data.iloc[2]['Description']
    SubRedditContent3 = sub_reddit_data.iloc[2]['Content']

    SubRedditName4 = sub_reddit_data.iloc[3]['SubReddit']
    SubRedditDes4 = sub_reddit_data.iloc[3]['Description']
    SubRedditContent4 = sub_reddit_data.iloc[3]['Content']

    SubRedditName5 = sub_reddit_data.iloc[4]['SubReddit']
    SubRedditDes5 = sub_reddit_data.iloc[4]['Description']
    SubRedditContent5 = sub_reddit_data.iloc[4]['Content']

    SubRedditName6 = sub_reddit_data.iloc[5]['SubReddit']
    SubRedditDes6 = sub_reddit_data.iloc[5]['Description']
    SubRedditContent6 = sub_reddit_data.iloc[5]['Content']

    SubReddit_List = []
    for index, row in tqdm(Agent_df.iterrows()):
            UserName = row['user_name']
            Gender = row['Gender']
            Age = row['Age']
            Region = row['Region']
            Education = row['EducationLevel']
            Ideology = row['Ideology']
            Description = row['Description']
            Event = row['Preference']
            SubRedditNum = row['SubRedditNum']

            Prompt_SubReddit = f"""You are {UserName}, your gender is {Gender}, your age is {Age}, you are from {Region}, your education level is {Education}, your political ideology is {Ideology}, your personal description is {Description}, and you are primarily interested in {Event} events. Below are descriptions of six SubReddits [SubRedditInfo], including the SubReddit name, the SubReddit description, and the SubReddit content. You need to analyze your personal information and select {SubRedditNum} SubReddits that interest you. Your [Response] must follow the [Format] format. The list length must be {SubRedditNum}. Please output only lists, no code or intermediate.

                [SubRedditInfo]: 

                    SubReddit 1:
                        SubReddit Name: {SubRedditName1}
                        SubReddit Description: {SubRedditDes1}
                        SubReddit Content: {SubRedditContent1}
                    SubReddit 2:
                        SubReddit Name: {SubRedditName2}
                        SubReddit Description: {SubRedditDes2}
                        SubReddit Content: {SubRedditContent2}
                    SubReddit 3:
                        SubReddit Name: {SubRedditName3}
                        SubReddit Description: {SubRedditDes3}
                        SubReddit Content: {SubRedditContent3}
                    SubReddit 4:
                        SubReddit Name: {SubRedditName4}
                        SubReddit Description: {SubRedditDes4}
                        SubReddit Content: {SubRedditContent4}
                    SubReddit 5:
                        SubReddit Name: {SubRedditName5}
                        SubReddit Description: {SubRedditDes5}
                        SubReddit Content: {SubRedditContent5}
                    SubReddit 6:
                        SubReddit Name: {SubRedditName6}
                        SubReddit Description: {SubRedditDes6}
                        SubReddit Content: {SubRedditContent6}
                    
                [Format]:
                    ['politics', 'InternationalNews', ...]
                
                [Response]:
            """
            i = 1
            while i:
                lists = generate(client, model, Prompt_SubReddit, temperature)
                if lists is None or len(lists) != SubRedditNum:
                    continue
                else:
                    SubReddit_List.append(lists) 
                    break
            if index == 500:
                csv_file_path = "./Data/LLMAgentProfile/SubReddit_List.csv"
                data = {
                        'SubReddit_List': SubReddit_List
                        }
                df = pd.DataFrame(data)
                df.to_csv(csv_file_path, index=False)

    csv_file_path = "./Data/LLMAgentProfile/agent_profiles.csv"
    df = pd.read_csv(csv_file_path)
    df['SubRedditList'] = SubReddit_List
    df.to_csv(csv_file_path, index=False)

main()