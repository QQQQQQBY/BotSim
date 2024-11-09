# Choose the most suitable LLMs

In order to select the most suitable macromodels, we conducted comprehensive tests on their commenting capabilities, with evaluation criteria covering the **relevance, diversity, and controllability of commented content**. 

Based on the results of these tests, we selected the top-performing LLMs **GPT4o-mini** to play a key role in the subsequent generation of the **BotSim-24 dataset**.

The LLMs we have tried include: **llama3-8b, chatglm3-6b, qwen1.5-72B, mistral-7b, gpt-3.5, and gpt4o-mini**.


## Post Data

**Original Posts**: We used different LLMs to comment to 18 original political posts to evaluate the diversity, controllability, and relevance of their comments.

## LLM-Select-Code

Execute the following command to run the code:

**`python LLM-Select/LLM-Select-Code/invoke_llm.py'**

## CommentData

Presentation of comments on different LLMs.

Six CSV files.