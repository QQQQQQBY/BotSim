import csv
import pandas as pd

prompt3 = """
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

prompt2 = """1. Read the content of [Post] and reply a comment.
2. The length of [Output] is independent of [Post].
3. The length of [Output] must approximate {Imitation_length}.
4. You must mimic the linguistic style, sentence length, and structure found in [Imitation].        
5. The format of [Output] must be a format showed in [Format].

        [Post]: {Post}
        [Imitation]: {Imitation}

        [Format]:
            [Output]: "text"
            
        [Output]:
            """
prompt1 = """Read the content of [Post] and reply a comment.    The format of [Output] must be a format showed in [Format].

        [Post]: {Post}

        [Format]:
            [Output]: "text"
            
        [Output]:
            """

def post_data():
    fb_posts = pd.read_excel("LLM-Select/Post-Data/Posts.xlsx", usecols=['Posts'])
    posts_list = fb_posts['Posts'].tolist() 
    imitate_comments = pd.read_excel("LLM-Select/Post-Data/imitations.xlsx", usecols=['comments'])
    imitate_comments = imitate_comments['comments'].tolist() 
    return posts_list, imitate_comments

def save_csv(output_list, model):
    csv_file = f'LLM-Select/CommentData/{model}_output_2.csv'
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = output_list[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_list)
        

