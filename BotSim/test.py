from action import *
from env import RedditEnv
from utils import load_config


config = load_config("config.yaml")
env = RedditEnv(config)
init = env.reset()
print(init)

action = CreateUserAction()
input_args = {
    "user_name": "iie test",
    "icon_img": "https://iietest.jpg",
    "time": "2024-02-16 00:00:00",
}
obs, agent = env.step(action, input_args, None)
print(obs)
action = PostAction()
input_args = {
    "subreddit": "python",
    "title": "iie test",
    "content_text": "iie test",
    "content_url": "https://iietest.jpg",
    "is_original_content": True,
    "is_self": False,
    "time": "2024-02-16 00:00:00",
}
print(env.step(action, input_args, agent))
action = CommentAction()
input_args = {
    "parent_id": "7jvdmtj",
    "comment_content": "new comment",
    "time": "2024-02-16 00:00:00",
}
print(env.step(action, input_args, agent))
action = LikeAction()
input_args = {"post_id": "7jvdmtj"}
print(env.step(action, input_args, agent))
action = DislikeAction()
input_args = {"post_id": "7jvdmtj"}
print(env.step(action, input_args, agent))

action = SearchPostAction()
input_args = {"keywords": "new"}
print(env.step(action, input_args, agent))
action = SearchUserAction()
input_args = {"keywords": "test"}
print(env.step(action, input_args, agent))
action = BrowseHotsInitPage()
input_args = {}
print(env.step(action, input_args, agent))

action = BrowseHotsSelectPage()
input_args = {"page_id": 5}
print(env.step(action, input_args, agent))
action = FollowAction()
input_args = {"user_id": "rgneo4c5"}
print(env.step(action, input_args, agent))
