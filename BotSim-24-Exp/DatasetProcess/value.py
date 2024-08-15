import pandas as pd
import ast 
import torch
import numpy as np

csv_file_path = "RedditBotSim/Data/RedditData/Posts.csv"

metadata = pd.read_csv(csv_file_path)


# username length
usernama_length = []
for na in metadata['name'].tolist():
    usernama_length.append(len(na))

# user posts number
user_posts_number = []
for pn in metadata['submission_num'].tolist():
    user_posts_number.append(int(pn))

# user comments number
user_comments_1_number = []
for cn in metadata['comment_num_1'].tolist():
    user_comments_1_number.append(int(cn))

# user comments 2 number
user_comments_2_number = []
for cn in metadata['comment_num_2'].tolist():
    user_comments_2_number.append(int(cn))

# user comments number
user_comments_number = []
for cn in metadata['comment_num'].tolist():
    user_comments_number.append(int(cn))

# user subreddit number
user_subreddit_number = []
for sn in metadata['subreddit'].tolist():
    sn = ast.literal_eval(sn)
    user_subreddit_number.append(len(sn))

# post comment 1 ratios
post_comment_1_ratios = []
for pn, cn1 in zip(user_posts_number, user_comments_1_number):
    if cn1 != 0:
        post_comment_1_ratios.append(pn/cn1)
    else:
        post_comment_1_ratios.append(0)

# post comment 2 ratios
post_comment_2_ratios = []
for pn, cn2 in zip(user_posts_number, user_comments_2_number):
    if cn2 != 0:
        post_comment_2_ratios.append(pn/cn2)
    else:
        post_comment_2_ratios.append(0)

# post comment ratios
post_comment_ratios = []
for pn, cn in zip(user_posts_number, user_comments_number):
    if cn != 0:
        post_comment_ratios.append(pn/cn)
    else:
        post_comment_ratios.append(0)

# post subreddit ratios 
post_subreddit_ratios = []
for pn, sn in zip(user_posts_number, user_subreddit_number):
    if sn != 0:
        post_subreddit_ratios.append(pn/sn)
    else:
        post_subreddit_ratios.append(0)


# regularization
usernama_length=pd.DataFrame(usernama_length)
usernama_length=(usernama_length-usernama_length.mean())/usernama_length.std()
usernama_length=torch.tensor(np.array(usernama_length),dtype=torch.float32)

user_posts_number = pd.DataFrame(user_posts_number)
user_posts_number = (user_posts_number-user_posts_number.mean())/user_posts_number.std()
user_posts_number = torch.tensor(np.array(user_posts_number),dtype=torch.float32)

user_comments_1_number = pd.DataFrame(user_comments_1_number)
user_comments_1_number = (user_comments_1_number-user_comments_1_number.mean())/user_comments_1_number.std()
user_comments_1_number = torch.tensor(np.array(user_comments_1_number),dtype=torch.float32)

user_comments_2_number = pd.DataFrame(user_comments_2_number)
user_comments_2_number = (user_comments_2_number-user_comments_2_number.mean())/user_comments_2_number.std()
user_comments_2_number = torch.tensor(np.array(user_comments_2_number),dtype=torch.float32)

user_comments_number = pd.DataFrame(user_comments_number)
user_comments_number = (user_comments_number-user_comments_number.mean())/user_comments_number.std()
user_comments_number = torch.tensor(np.array(user_comments_number),dtype=torch.float32)

user_subreddit_number = pd.DataFrame(user_subreddit_number)
user_subreddit_number = (user_subreddit_number-user_subreddit_number.mean())/user_subreddit_number.std()
user_subreddit_number = torch.tensor(np.array(user_subreddit_number),dtype=torch.float32)

post_comment_1_ratios = pd.DataFrame(post_comment_1_ratios)
post_comment_1_ratios = (post_comment_1_ratios-post_comment_1_ratios.mean())/post_comment_1_ratios.std()
post_comment_1_ratios = torch.tensor(np.array(post_comment_1_ratios),dtype=torch.float32)

post_comment_2_ratios = pd.DataFrame(post_comment_2_ratios)
post_comment_2_ratios = (post_comment_2_ratios-post_comment_2_ratios.mean())/post_comment_2_ratios.std()
post_comment_2_ratios = torch.tensor(np.array(post_comment_2_ratios),dtype=torch.float32)

post_comment_ratios = pd.DataFrame(post_comment_ratios)
post_comment_ratios = (post_comment_ratios-post_comment_ratios.mean())/post_comment_ratios.std()
post_comment_ratios = torch.tensor(np.array(post_comment_ratios),dtype=torch.float32)

post_subreddit_ratios = pd.DataFrame(post_subreddit_ratios)
post_subreddit_ratios = (post_subreddit_ratios-post_subreddit_ratios.mean())/post_subreddit_ratios.std()
post_subreddit_ratios = torch.tensor(np.array(post_subreddit_ratios),dtype=torch.float32)

num_properties_tensor=torch.cat([usernama_length,user_posts_number,user_comments_1_number,user_comments_2_number,user_comments_number, user_subreddit_number, post_comment_1_ratios, post_comment_2_ratios, post_comment_ratios, post_subreddit_ratios],dim=1)

path = "BotSim-24-Exp/Dataset/"
torch.save(num_properties_tensor,path+'num_properties_tensor.pt')