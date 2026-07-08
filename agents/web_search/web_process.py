import os
from web_search import web_search
from typing import Dict , List
from dotenv import load_dotenv

class web_process:
    def __init__(self , config):
        self.web_process = web_search()
        self.llm = config.web_search.llm
    
    def _create_prompt(self , query , chat_history):
        prompt = f"""Here are the last few messages from our conversation:

        {chat_history}

        The user asked the following question:

        {query}

        Summarize them into a single, well-formed question only if the past conversation seems relevant to the current query so that it can be used for a web search.
        Keep it concise and ensure it captures the key intent behind the discussion.
        """

        return prompt
    
    def process_web_result(self , query , chat_history):
        prompt = self._create_prompt(query=query , chat_history=chat_history)

        web_search_query = self.llm.invoke(prompt)

        web_result = self.web_process.web_search(web_search_query.content)

        llm_prompt = (
            "You are an AI assistant specialized in medical information. Below are web search results "
            "retrieved for a user query. Summarize and generate a helpful, concise response. "
            "Use reliable sources only and ensure medical accuracy.\n\n"
            f"Query: {query}\n\nWeb Search Results:\n{web_result}\n\nResponse:"
        )

        response = self.llm.invoke(llm_prompt)

        return response

