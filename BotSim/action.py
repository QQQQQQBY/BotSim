from basic import Action
from datetime import datetime

from typing import List, Optional, Callable, Dict, Any, Type


# class BrowsePost(Action):

#     def __init__(self):
#         super().__init__(
#             name="BrowsePost",
#             description="Browse a post.",
#             func=self.browse_post,
#             input_args_schema={"post_id": int},
#         )

#     @staticmethod
#     def browse_post(action_args: Dict[str, Any], env, agent):
#         return env.get_env_reddit_data().get_all_post_content()


class BrowseRecommendPostInitPage(Action):
    def __init__(self):
        super().__init__(
            name="BrowseRecommendPostInitPage",
            description="""Meaning: Going through personalized content suggestions based on the your history and interests.
                Execution: Engage with this when wanting to discover content tailored to the your preferences.
                """,
            func=self.browse_recommend,
            input_args_schema={},
        )

    @staticmethod
    def browse_recommend(action_args, env, agent):
        feed_posts = env.get_env_reddit_data.get(agent.name)
        if feed_posts is None:
            return ""
        read_rows = feed_posts.sample(n=env.feed_num)
        feed_posts.drop(read_rows.index, inplace=True)

        result = []
        for i, row in read_rows.iterrows():
            result.append("Feed Article id {}: {}".format(row["id"], row["text"][:600]))
        return "\n\n".join(result)


class BrowseRecommendPostSelectPage(Action):
    def __init__(self):
        super().__init__(
            name="BrowseRecommendPostSelectPage",
            description="""Meaning: Going through personalized content suggestions based on the your history and interests.
                Execution: Engage with this when wanting to discover content tailored to the your preferences.
                """,
            func=self.browse_recommend,
            input_args_schema={"page_id": int},
        )

    @staticmethod
    def browse_recommend(action_args, env, agent):
        feed_posts = env.feed_posts.get(agent.name)
        if feed_posts is None:
            return ""
        read_rows = feed_posts.sample(n=env.feed_num)
        feed_posts.drop(read_rows.index, inplace=True)

        result = []
        for i, row in read_rows.iterrows():
            result.append("Feed Article id {}: {}".format(row["id"], row["text"][:600]))
        return "\n\n".join(result)


class BrowseHotsInitPage(Action):

    def __init__(self):
        super().__init__(
            name="BrowseHotsInitPage",
            description="""Meaning: Viewing posts that are currently trending or popular in a specific subreddit or Reddit as a whole. 
            Parameter meanings: None. 
            Execution: Browse these posts when aiming to stay updated with the latest popular content.""",
            func=self.browse_hot,
            input_args_schema={},
        )

    @staticmethod
    def browse_hot(action_args: Optional[Dict[str, Any]], env, agent):
        post_list_by_hot_value_on_init_page = (
            env.get_env_reddit_data().get_post_list_content_by_hot_value_page(
                page_id=0, agent=agent
            )
        )
        cur_page = 0
        post_count, page_size, max_page = (
            env.get_env_reddit_data().get_post_list_number_pagesize_maxpage_by_hot_value()
        )
        return f"Post Count: {post_count}, Page Size: {page_size}, Max Page: {max_page}, Page {cur_page} Content: {post_list_by_hot_value_on_init_page}"


class BrowseHotsSelectPage(Action):

    def __init__(self):
        super().__init__(
            name="BrowseHotsSelectPage",
            description="""Meaning: Viewing posts that are currently trending or popular in a specific subreddit or Reddit as a whole.
                Parameter meanings: [page_id: the page number of the trending or popular posts to browse.]
                Execution: Browse these posts when aiming to stay updated with the latest popular content, specifying the page number to navigate through the results.""",
            func=self.browse_hot,
            input_args_schema={"page_id": int},
        )

    @staticmethod
    def browse_hot(action_args: Optional[Dict[str, Any]], env, agent):
        post_list_by_hot_value_on_select_page = (
            env.get_env_reddit_data().get_post_list_content_by_hot_value_page(
                page_id=(action_args["page_id"] - 1), agent=agent
            )
        )
        if post_list_by_hot_value_on_select_page:
            cur_page = str(action_args["page_id"])
            post_count, page_size, max_page = (
                env.get_env_reddit_data().get_post_list_number_pagesize_maxpage_by_hot_value()
            )
            return f"Post Count: {post_count}, Page Size: {page_size}, Max Page: {max_page}, Page {cur_page} Content: {post_list_by_hot_value_on_select_page}"
        return "Page Id Is Wrong!"


class BrowseRisings(Action):
    pass


class SearchPostAction(Action):
    def __init__(self):
        super().__init__(
            name="SearchPost",
            description="""Meaning: Looking up specific posts using keywords.
                Parameter meanings: [keywords: the keywords used to search for the specific post.].
                Execution: Use when you're seeking specific information or topics not readily available in your feed.""",
            func=self.search,
            input_args_schema={"keywords": str},
        )

    @staticmethod
    def search(action_args: Optional[Dict[str, Any]], env, agent):
        observation = env.get_env_reddit_data().search_post_by_keywords(
            keywords=action_args["keywords"]
        )
        return observation


class SearchUserAction(Action):
    def __init__(self):
        super().__init__(
            name="SearchUser",
            description="""Meaning: Looking up specific user using keywords.
                Parameter meanings: [keywords: the keywords used to search for the specific user.].
                Execution: Use when you're seeking specific information or topics not readily available in your feed.""",
            func=self.search,
            input_args_schema={"keywords": str},
        )

    @staticmethod
    def search(action_args: Optional[Dict[str, Any]], env, agent):
        observation = env.get_env_reddit_data().search_user_by_keywords(
            keywords=action_args["keywords"]
        )
        return observation


class PostAction(Action):

    def __init__(self):
        super().__init__(
            name="PostAction",
            description="""Meaning: Sharing original content or links to the Reddit platform in the form of text, image, video, or a link.
                Parameter meanings: [subreddit: the name of the subreddit to which the post belongs, 
                title: providing a brief overview of its main content,
                content_text: the main body of text for text-based posts or a description for link posts,
                content_url: the URL for posts that are link shares or media uploads,
                is_original_content: a flag indicating whether the post contains original content created by the user,
                is_self: a flag indicating whether the post is a selfpost(text-only),
                time: the time to execute this action,  the time at which the action is to be executed. The format of the time parameter is similar to "2024-02-15 00:00:00". It is crucial that the action is executed within the time frame defined by the system's start_time and end_time.].
                Execution: Ensure content is relevant, original, and adheres to the guidelines of the specific subreddit.""",
            func=self.post,
            input_args_schema={
                "subreddit": str,
                "title": str,
                "content_text": str,
                "content_url": str,
                "is_original_content": bool,
                "is_self": bool,
                "time": str,
            },
        )

    @staticmethod
    def post(action_args: Optional[Dict[str, Any]], env, agent):
        observation = (
            env.get_env_reddit_data().add_post_by_subreddit_title_content_agent(
                subreddit=action_args["subreddit"],
                title=action_args["title"],
                content_text=action_args["content_text"],
                content_url=action_args["content_text"],
                is_original_content=action_args["is_original_content"],
                is_self=action_args["is_self"],
                time=action_args["time"],
                agent=agent,
            )
        )
        return observation


class CommentAction(Action):

    def __init__(self):
        super().__init__(
            name="Comment",
            description="""Meaning: Responding or adding input to a post or another comment.
                Parameter meanings: [parent_id: the identifier of the parent post or comment to which the comment belongs, 
                comment_content: the content of the comment, containing valuable insights, questions, or responses,
                time: the time to execute this action,  the time at which the action is to be executed. The format of the time parameter is similar to "2024-02-15 00:00:00". It is crucial that the action is executed within the time frame defined by the system's start_time and end_time.]
                Execution: Comment when you have valuable insights, questions, or responses to contribute.
                """,
            func=self.comment,
            input_args_schema={"parent_id": str, "comment_content": str, "time": str},
        )

    @staticmethod
    def comment(action_args: Optional[Dict[str, Any]], env, agent):
        observation = env.get_env_reddit_data().add_comment_by_parent_id_content_agent(
            parent_id=action_args["parent_id"],
            content=action_args["comment_content"],
            time=action_args["time"],
            agent=agent,
        )
        return observation


class LikeAction(Action):
    def __init__(self):
        super().__init__(
            name="Like",
            description="""Meaning: Expressing approval or appreciation for a particular post or comment.
                Parameter meanings: [post_id: the identifier of the post to which the like is directed,
                time: the time to execute this action,  the time at which the action is to be executed. The format of the time parameter is similar to "2024-02-15 00:00:00". It is crucial that the action is executed within the time frame defined by the system's start_time and end_time.]
                Execution: Upvote when you find content insightful, valuable, or agreeable.""",
            func=self.like,
            input_args_schema={"post_id": str},
        )

    @staticmethod
    def like(action_args: Optional[Dict[str, Any]], env, agent):
        observation = env.get_env_reddit_data().like_post_by_post_id(
            post_id=action_args["post_id"]
        )
        return observation


class DislikeAction(Action):
    def __init__(self):
        super().__init__(
            name="Dislike",
            description="""Meaning: Expressing disapproval or disagreement with content.
                Parameter meanings: [post_id: he identifier of the post to which the dislike is directed.]
                Execution: Downvote only when content is not relevant, misleading, or violates community norms. """,
            func=self.dislike,
            input_args_schema={"post_id": str},
        )

    @staticmethod
    def dislike(action_args: Optional[Dict[str, Any]], env, agent):
        observation = env.get_env_reddit_data().dislike_post_by_post_id(
            post_id=action_args["post_id"]
        )
        return observation


class FollowAction(Action):
    def __init__(self):
        super().__init__(
            name="Follow",
            description="""Meaning: Subscribing to updates from a user.
                Parameter meanings: [user_id: The unique identifier of the user to follow.]
                Execution: Initiate following to receive notifications and updates from the specified user.""",
            func=self.follow,
            input_args_schema={"user_id": str},
        )

    @staticmethod
    def follow(action_args: Optional[Dict[str, Any]], env, agent):
        observation = env.get_env_reddit_data().follow_user_by_user_id(
            user_id=action_args["user_id"], agent=agent
        )
        return observation


class CreateUserAction(Action):
    def __init__(self):
        super().__init__(
            name="CreateUser",
            description="""Meaning: Creating a new user profile with specified information.
                Parameter meanings: [user_name: the username for the new user profile,
                icon_img: a URL link to the profile icon image that represents the new user,
                time: the time to execute this action,  the time at which the action is to be executed. The format of the time parameter is similar to "2024-02-15 00:00:00". It is crucial that the action is executed within the time frame defined by the system's start_time and end_time.].
                Execution: Execute this action to create a new user profile with the provided username and description. """,
            func=self.add_agent,
            input_args_schema={"user_name": str, "icon_img": str, "time": str},
        )

    @staticmethod
    def add_agent(action_args: Optional[Dict[str, Any]], env, agent):
        observation = env.get_env_reddit_data().add_user_by_user_name_description(
            user_name=action_args["user_name"],
            icon_img=action_args["icon_img"],
            time=action_args["time"],
        )
        return observation


class StopAction(Action):
    def __init__(self):
        super().__init__(
            name="Stop",
            description="",
            func=self.stop,
            input_args_schema={},
        )

    @staticmethod
    def stop(action_args: Optional[Dict[str, Any]], env, agent):
        observation = "Something goes wrong, stop this action"
        return observation