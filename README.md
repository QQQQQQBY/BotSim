<h1 align="center"> 👾 BotSim 👾</h1>


<h3 align="center">
    <p>LLM-Powered Malicious Social Botnet Simulation</p>
</h3>


 <p align="center">
<img src="./imgs/botnet.png" width="400">
</p>

<p align="center">
    [<a href="https://arxiv.org/pdf/2412.13420">Paper</a>]
</p>

**BotSim** is a simulation framework designed to emulate the participation of social bots in information dissemination. This framework integrates both regular user accounts and malicious bot accounts, encompassing common interactive behaviors found on social media platforms, such as liking, commenting, and posting. Furthermore, BotSim is equipped with an intelligent recommendation strategy function that enables precise message stream recommendation for each bot account, thereby achieving a highly realistic simulation of the dynamics of information propagation within social networks.

- User Network Construction: By collecting user behavior data from real social media platforms, the system constructs a highly realistic social network graph. The relationships between user nodes can be flexibly configured based on the data collection strategy, enabling the construction of various networks such as follower-following relationships, reposting networks, and reply networks.


- Bot User Modeling: Utilizing a programmatic control mechanism, this approach constructs a multidimensional attribute set for bot users, encompassing personal information and social behavior characteristics. Algorithms are employed to facilitate natural interactions with real user accounts.


- Action List: A set of information interaction behaviors has been constructed, encompassing action modules such as posting, reposting, liking, and commenting, which supports the simulation of social dissemination processes.


- Recommendation Function: By comprehensively considering features such as post publication time and interaction popularity (such as number of likes), it enables the recommendation of information streams, simulating the behavior of recommendation systems on real social platforms.

 <p align="center">
<img src="./imgs/modeloverview.jpg" width="600">
</p>



# 🚀 Getting Started

## 📰 BotSim folder

- The BotSim folder contains the code for constructing the BotSim simulation. To run the simulation code, execute `python test.py`.

- Data preparation: Prepare the data format required for the program to run.

- Due to the loss of the previous code, this part of the code uploaded in the github version is not complete, and we will update the code that can be fully reproduced in the future.

- Currently, the network functionality module simulating information dissemination on the Reddit platform is running stably. For experiment replication, it is recommended to refer to the instruction document in the **RedditBotSim** folder for testing.

## ✍️ LLM-Select folder

- The LLM-Select folder introduces the LLMs selection strategy used in the construction of the BotSim-24 dataset is described in detail.

- You can view the detailed readme file in the LLM-Select folder. [<a href="LLM-Select/readme.md">readme</a>]


## ✨ RedditBotSim folder

The RedditBotSim folder includes the code for building the BotSim-24 dataset within the Reddit environment. To run the dataset construction code, execute `python ./AgentDecisionCenter/main.py`.

- First, our framework is powered by GPT-4o-mini. To ensure proper operation, please assign values to `openai_api_base=''` and `openai_api_key=''`. <span style="color:grey;font-size:10px;">(RedditBotSim/AgentDesicionCenter/main.py & RedditBotSim/AgentDesicionCenter/modify_content.py & RedditBotSim/Action/CreateAgentBots.py)</span>

- Second, we first counted the number of real users' posts, post community, comment community and other basic information. After that, we based on the LLM and automation program for manipulating bot generates these information, relevant code is: `RedditBotSim/Action/CreateAgentBots.Py`

- Third, install the necessary python packages, perform ` python. / AgentDecisionCenter/main py ` to run the program.


## 💡 BotSim-24-Dataset folder

- This is our BotSim-24 dataset based on the RedditBotSim project.

- You can view the detailed readme file in the BotSim-24-Dataset folder. [<a href="BotSim-24-Dataset/Readme.md">readme</a>]


## 🌟 BotSim-24-Exp folder

- The BotSim-24-Exp folder showcases the detection performance benchmarks for bot detection on the BotSim-24 dataset. The code for different methods is integrated into a single file, which can be run directly.

- The encoded features in the BotSim-24-mini-sample folder can be used to reproduce the results.

## ⚙️ BotSim-24-mini-sample folder

- The BotSim-24-mini-sample folder presents a subset of the dataset, including profile information (`metadata.csv`) and text data (`text.json`).

- We also show the coding features required for bot detection experiments.

## 🛠 Background Knowledge Data

- Code for collecting background knowledge data: [Code Link](https://github.com/QQQQQQBY/CrawlNYTimes).

- The background knowledge dataset is available at [Google Drive](https://drive.google.com/drive/folders/16zS_Gq45ckeixeW9JQZbi71TYPFSwj5X?usp=drive_link).


## 🥳 Citation
- Our work has been accepted by AAAI2025. If you find this repo helpful, feel free to cite us.

```
@article{qiao2024botsim,
  title={BotSim: LLM-Powered Malicious Social Botnet Simulation},
  author={Qiao, Boyu and Li, Kun and Zhou, Wei and Li, Shilong and Lu, Qianqian and Hu, Songlin},
  journal={arXiv preprint arXiv:2412.13420},
  year={2024}
}
```