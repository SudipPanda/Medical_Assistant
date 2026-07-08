from typing import List , Dict , Any , Optional
from web_process import web_process


class Web_Process_Val:
    def __init__(self):
        self.web_process_val = web_process()
    
    def process_web_search(self , query:str, chat_history)->str:
        return self.web_process_val.process_web_result(query , chat_history)
