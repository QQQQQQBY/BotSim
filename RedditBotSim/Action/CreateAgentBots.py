# (1)  Since 1000 Agent Bots need to be constructed, instead of using Prompt to construct Agents in Appendix B.3, 1000 Agents are constructed in batch using the behavior of `Create User Action', and then LLM is applied subsequently to plan actions for each Agent.

# UserName; Age; Gender; Job; Education Level; MBTI; Hobbies; Region; Language; UserDescription
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
# Generate UserName; Description
def generate(client, model, prompt, temperature, keyword_list, keyword):
    completion = client.chat.completions.create(
                        model = model,
                        messages = [{"role": "user", "content": prompt}],
                        temperature= temperature
                    )
    llm_return_dict = completion.choices[0].message.content
    try:
        llm_return_dict = json.loads(llm_return_dict)
        for k in llm_return_dict:
            if k[keyword] in keyword_list:
                llm_return_dict = generate(client, model, prompt, temperature, keyword_list, keyword)

    except Exception as e:
            print("======Exception====== \n", e)
            llm_return_dict = generate(client, model, prompt, temperature, keyword_list, keyword)
    return llm_return_dict

# # Age, Gender
def generate_age_gender_Region_edu_ideo(client, model, prompt, temperature, Num):
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
            # if len(lists) != Num or len(lists) == 0 or lists == None:
            #     return None
            # if len(lists) == Num and len(lists) > 0:
            if isinstance(lists, list): 
                return lists
            else:
                return None
        else: 
            return None
    except Exception as e:
        print("======Exception====== /n", e)
        return None



def generate_reddit_id(existing_ids, length=7):
    characters = string.ascii_lowercase + string.digits
    existing_ids_set = list(set(existing_ids))

    reddit_id = "".join(random.choice(characters) for _ in range(length))
    if reddit_id not in existing_ids_set:
        existing_ids_set.append(reddit_id)
        return reddit_id, existing_ids_set
    else: 
        reddit_id, existing_ids_set = generate_reddit_id(existing_ids_set, length=7)
    

def main():
    
    model='gpt-4o-mini'
    temperature=0.9
    openai_api_base=''
    
    openai_api_key=''


    
    client = OpenAI(
            api_key = openai_api_key,
            base_url = openai_api_base
            )

    # Age
    AgeNum = 20
    Prompt_age = f"""[Task]:You are asked to generate {AgeNum} age data. The proportion of ages must reflect the statistics for provided [Statistics].  You can write code to assist with the generation of this data. The [Format] of your [Response] needs to be a list, and the list length must be {AgeNum}. Please output only lists, no code or intermediate results.


    [Statistics]:
        44% of users are in the 18 to 29 age.
        31% of users are in the 30 to 49 age.
        11% of users are in the 50 to 64 age.
        3% of users are in the 65 to 70 age.

    [Format]:
        [23, 45, ...]

    [Response]: """

    Age_List = []
    i = 0
    while i < 50:
        lists = generate_age_gender_Region_edu_ideo(client, model, Prompt_age, temperature, AgeNum)
        if lists is None:
            continue  
        else:
            Age_List.append(lists)
            i += 1  


    # Gender
    GenderNum = 20
    Prompt_gender = f"""[Task]:You are asked to generate {GenderNum} gender data. The proportion of genders must reflect the statistics for provided [Statistics]. You can write code to assist with the generation of this data. The [Format] of your [Response] needs to be a list, and the list length must be {GenderNum}. Please output only lists, no code or intermediate results.

    [Statistics]:
        'Female' users account for 35.1%.
        'Male' users account for 63.6%.

    [Format]:
        ['Female', 'Male', ...]

    [Response]: """

    Gender_List = []
    i = 0
    while i < 50:
        lists = generate_age_gender_Region_edu_ideo(client, model, Prompt_gender, temperature, GenderNum)
        if lists is None:
            continue  
        else:
            Gender_List.append(lists)
            i += 1  
        



    # Region
    RegionNum = 20
    Prompt_region = f"""[Task]:You are asked to generate {RegionNum} region data in US. The proportion of regions must reflect the statistics for provided [Statistics]. You can write code to assist with the generation of this data. The [Format] of your [Response] needs to be a list, and the list length must be {RegionNum}. Please output only lists, no code or intermediate results. 

    [Format]:
        ['American, New York City', 'American, Los Angeles', ...]

    [Response]: """

    Region_List = []
    i = 0
    while i < 50:
        lists = generate_age_gender_Region_edu_ideo(client, model, Prompt_region, temperature, RegionNum)
        if lists is None:
            continue  
        else:
            Region_List.append(lists)
            i += 1  

    # Education Level
    EducationNum = 20
    Prompt_education = f"""[Task]:You are asked to generate {EducationNum} education level data. The proportion of education levels must reflect the statistics for provided [Statistics]. You can write code to assist with the generation of this data. The [Format] of your [Response] needs to be a list, and the list length must be {EducationNum}. Please output only lists, no code or intermediate results. 

    [Statistics]:
        46% holding at least a college degree or higher.
        40% have a high school diploma. 
        14% others.

    [Format]:
        ['Bachelor's Degree in Cultural Anthropology', 'Ph.D. in Art History', ...]

    [Response]: """

    Education_List = []
    i = 0
    while i < 50:
        lists = generate_age_gender_Region_edu_ideo(client, model, Prompt_education, temperature, EducationNum)
        if lists is None:
            continue  
        else:
            Education_List.append(lists)
            i += 1  


    
    # Ideology on social issues
    IdeologyNum = 20
    Prompt_Ideology = f"""[Task]:You are asked to generate {IdeologyNum} ideology data. The proportion of ideology must reflect the statistics for provided [Statistics]. You can write code to assist with the generation of this data. The [Format] of your [Response] needs to be a list, and the list length must be {IdeologyNum}. Please output only lists, no code or intermediate  

    [Statistics]:
        'Conservative' account for 32%.
        'Moderate' account for 32%.
        'Liberal' account for 33%.

    [Format]:
        ['Conservative', 'Moderate', 'Liberal', ...]

    [Response]: """

    Ideology_List = []
    i = 0
    while i < 50:
        lists = generate_age_gender_Region_edu_ideo(client, model, Prompt_Ideology, temperature, IdeologyNum)
        if lists is None:
            continue  
        else:
            Ideology_List.append(lists)
            i += 1  


    # UserName
    Prompt_username = """[Task]:
        You need to generate 10 sample based on [Example]. "UserName" can only contain letters, numbers, "-" and "_". The format of [Response] must be a format such as showed in [Format].

    [Example]:

        [{"UserName": "CINDERELLA"}, {"UserName": "Theodore Bobcat for life"}, {"UserName": "Severe_County_5041"}, {"UserName": "Yan"}, {"UserName": "juniperblossomss"}, {"UserName": "Forsaken-Duck-8142"}, {"UserName": "Gintin2"}, {"UserName": "Bluerecyclecan"}, {"UserName": "JussiesTunaSub"}, {"UserName": "TanEnojadoComoTu"}, {"UserName": "TheDarthSnarf"} ]
    
    [Format]:
        [{"UserName": "CINDERELLA"}, {"UserName": "Theodore Bobcat for life"}]

    [Response]: """

    username_list = []
    
    for i in tqdm(range(100)):
        llm_return_dict = generate(client, model, Prompt_username, temperature, username_list, "UserName")

        for name in llm_return_dict:
            username_list.append(name["UserName"])

        # username_list = list(set(username_list))

    existing_ids_set = []
    for j in range(1000):
        _, existing_ids_set = generate_reddit_id(existing_ids_set, length=7)

    # UserDescription
    Prompt_description = """[Task]:
        You are a Reddit user interested in international news. You must imitate the tone, sentence structure, etc. of [Example] to generate 10 samples. The format of [Response] must be a format such as showed in [Format].
        
    [Example]:
        [{"Description": "Be brave, be modest."}, {"Description": "Bubsy fans and enjoyers welcome here: https://discord.gg/rSMQs29HyZ"},{"Description": "swiftie (delusional)"} {"Description": "Hii, hope you have a good day : )"},{"Description": "Luhansk - Kyiv - Ukraine"} {"Description": "I like reading the news a lot."}, {"Description": "I'm not a bot, I'm just interested in Russia"}, {"Description": "there is no rhyme of reason to this account, I'm just here using Reddit like you are, you are free to block or follow"}, {"Description": "reporting from Kharkiv, Ukraine; grumpy editor"}]

    [Format]:
        [{"Description": "Be brave, be modest."}, {"Description": "Hii, hope you have a good day : )"}]

    [Response]: """

    description_list = []
    for i in tqdm(range(100)):

        llm_return_dict = generate(client, model, Prompt_description, temperature, description_list, "Description")
        
        for description in llm_return_dict:
            description_list.append(description["Description"])

    import csv
    import pandas as pd
    Age_List = list(filter(lambda x: x is not None, Age_List))
    Age_List = list(itertools.chain(*Age_List))

    Gender_List = list(filter(lambda x: x is not None, Gender_List))
    Gender_List = list(itertools.chain(*Gender_List))

    Region_List = list(filter(lambda x: x is not None, Region_List))
    Region_List = list(itertools.chain(*Region_List))

    Education_List = list(filter(lambda x: x is not None, Education_List))
    Education_List = list(itertools.chain(*Education_List))

    Ideology_List = list(filter(lambda x: x is not None, Ideology_List))
    Ideology_List = list(itertools.chain(*Ideology_List))

    ID_len = len(existing_ids_set)
    name_len = len(username_list)
    description_len = len(description_list)
    age_len = len(Age_List)
    gender_len = len(Gender_List)
    region_len = len(Region_List)
    education_len = len(Education_List)
    ideo_len = len(Ideology_List)
    min_len = min([ID_len, name_len, description_len, age_len, gender_len, region_len, education_len, ideo_len])

    # Specify the CSV file path
    csv_file_path = "./RedditBotSim/Data/LLMAgentProfile/agent_profiles.csv"

    

    # Preference distribution
    Preference_List = ['Russia-Ukraine war', 'Israeli-Palestinian conflict', 'US politics']

    preferences_count = {
    'Russia-Ukraine war': 250,
    'Israeli-Palestinian conflict': 201,
    'US politics': 549
    }

    # Generate data list
    data = []
    for preference, count in preferences_count.items():
        data.extend([preference] * count)

    Preference = random.shuffle(data)

    # SubRedditNum
    numbers = [1, 2, 3, 4, 5]
    weights = [602, 685, 487, 125, 8]
    SubRedditNum_List = random.choices(numbers, weights=weights, k=1000)

    # Write the lists into the CSV file
    with open(csv_file_path, mode='a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "user_name", "description", "Age", "Gender", "Region", "EducationLevel", "Ideology", "Preference", "SubRedditNum"])
        for item1, item2, item3, item4, item5, item6, item7,  item8, item9, item10 in zip(existing_ids_set[:min_len], username_list[:min_len], description_list[:min_len], Age_List[:min_len], Gender_List[:min_len], Region_List[:min_len], Education_List[:min_len], Ideology_List[:min_len], Preference[:min_len], SubRedditNum_List[:min_len]):
            writer.writerow([item1, item2, item3, item4, item5, item6, item7, item8, item9,item10])
    print(f"CSV file has been saved to: {csv_file_path}")
main()