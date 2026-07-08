import logging
from typing import Any , List , Dict , Optional

class ResponseGenerator:

    def __init__(self , config):
        self.logger = logging.getLogger(__name__)
        self.response_generator_model = config.rag.response_generator_model
        self.include_sources = getattr(config.rag, "include_sources", True)

    def _build_prompt(self , query: str, 
            context: str,
            chat_history: Optional[List[Dict[str, str]]] = None):
        
        table_instructions = """
        Some of the retrieved information is presented in table format. When using information from tables:
        1. Present tabular data using proper markdown table formatting with headers, like this:
            | Column1 | Column2 | Column3 |
            |---------|---------|---------|
            | Value1  | Value2  | Value3  |
        2. Re-format the table structure to make it easier to read and understand
        3. If any new component is introduced during re-formatting of the table, mention it explicitly
        4. Clearly interpret the tabular data in your response
        5. Reference the relevant table when presenting specific data points
        6. If appropriate, summarize trends or patterns shown in the tables
        7. If only reference numbers are mentioned and you can fetch the corresponding values like research paper title or authors from the context, replace the reference numbers with the actual values
        """

        response_format_instructions = """Instructions:
        1. Answer the query based ONLY on the information provided in the context.
        2. If the context doesn't contain relevant information to answer the query, state: "I don't have enough information to answer this question based on the provided context."
        3. Do not use prior knowledge not contained in the context.
        5. Be concise and accurate.
        6. Provide a well-structured response with heading, sub-headings and tabular structure if required in markdown format based on retrieved knowledge. Keep the headings and sub-headings small sized.
        7. Only provide sections that are meaningful to have in a chatbot reply. For example, do not explicitly mention references.
        8. If values are involved, make sure to respond with perfect values present in context. Do not make up values.
        9. Do not repeat the question in the answer or response."""

        prompt = f"""You are a medical assistant providing accurate information based on verified medical sources.

        Here are the last few messages from our conversation:
        
        {chat_history}

        The user has asked the following question:
        {query}

        I've retrieved the following information to help answer this question:

        {context}

        {table_instructions}

        {response_format_instructions}

        Based on the provided information, please answer the user's question thoroughly but concisely.
        If the information doesn't contain the answer, acknowledge the limitations of the available information.

        Do not provide any source link that is not present in the context. Do not make up any source link.

        Medical Assistant Response:"""

        return prompt
    
    def generate_response(
            self , 
             query: str,
            retrieved_docs: List[Dict[str, Any]],
            picture_paths: List[str],
            chat_history: Optional[List[Dict[str, str]]] = None,
       ):
        
        
        pass
            
        