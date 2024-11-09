import os
from openai import OpenAI
import random
import csv
from Utils.utils import *
from tqdm import tqdm
def generate(client, template, Imitation_length, count):
    messages=[
        {"role": "user", "content": template}
    ]
    try:
        response = client.chat.completions.create(
            model="mistral-7b-instruct",
            messages=messages
        )
        output = response.choices[0].message.content
    except Exception as e:        
        output = generate(client, template, Imitation_length, count)

    return output

def generate_comment(client, posts_list, imitate_comments):
    output_list = []
    for i in tqdm(range(len(posts_list))):
        Post = posts_list[i]
        random.seed(i)
        random_number = random.randint(0, 20)
        Imitation = imitate_comments[random_number]
        Imitation_length = len(Imitation)
        template = f"""
            1. Read the content of [Post] and reply a comment.
            2. The length of [Output] is independent of [Post].
            3. The length of [Output] must approximate {Imitation_length}.
            4. You must mimic the linguistic style, sentence length, and structure found in [Imitation].
            5. Your comment must disagree with the [Post] point of view.       
            6. The format of [Output] must be a format showed in [Format].

                    [Post]: {Post}
                    [Imitation]: {Imitation}

                    [Format]:
                        [Output]: "text"
                        
                    [Output]:
            """
        
        output = generate(client, template, Imitation_length, 0) 
        output_list.append({"Post": Post, "Imitation": Imitation, "Output": output})
    return output_list


if __name__ == '__main__':
    client = OpenAI(
    api_key = "",
    base_url = ""
    )
    posts_list, imitate_comments = post_data()
    output_list = generate_comment(client, posts_list, imitate_comments)
    
    save_csv(output_list, "mistral-7b-instruct")

#  "mistral-7b-instruct", "llama3-8b", "Qwen1.5-72B-Chat", "gpt-3.5-turbo-ca", "gpt4o-mini"