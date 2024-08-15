# Rewrite a knowledge
import json
import random
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import pandas as pd
import os
from openai import OpenAI
from datetime import datetime

def generate_news(client, model, prompt, temperature):
    try:
        completion = client.chat.completions.create(
                        model = model,
                        messages = [{"role": "user", "content": prompt}],
                        temperature= temperature
                    )
        llm_return_dict = completion.choices[0].message.content
        return llm_return_dict
    except Exception as e:
        print("======Exception====== /n", e)
        return None
        


def modify_content(Claim):

    model='gpt-4o'
    temperature=0.9
    openai_api_base=''
    openai_api_key=''

    input_llm_prompt = """Rewrite the news in [News] so that the generated news has a different point of view than the original news. You need to synthesize (1),(2),(3),(4),and (5) of the information to complete the response.
    (1) Modify key factors in the news, such as time, place, event, mood, opinion, etc.
    (2) You cannot directly quote the original [news], you only need to generate your revised 
    (3) The revised news [Response] should be logically consistent.
    (4) Please think step by step, how you can modify this [News].
    (5) The news data format you generate should be the same as the original news.
    [News]: {claim}
    [Response]:"""

    client = OpenAI(
            api_key = openai_api_key,
            base_url = openai_api_base
            )
    prompt = input_llm_prompt.format(claim=Claim)
    i = 1
    while i:
        llm_return_dict = generate_news(client, model, prompt, temperature)
        if llm_return_dict != None:
            break
        else:
            continue
    return llm_return_dict



