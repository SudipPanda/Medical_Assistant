import json
from dotenv import load_dotenv
from agents.rag_agent import Medical_Rag
from agents.web_search import Web_Process_Val 
from langchain_core.messages import HumanMessage , AIMessage ,SystemMessage , BaseMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Optional , Dict , Union , TypedDict
from langchain_core.prompts import PromptTemplate , ChatPromptTemplate

#from agents.image_preprocess import 
#from agents . gurdrails import 

from config import Config

load_dotenv()
Config = Config()

memory = MemorySaver()
thread_config = {"configurable": {"thread_id": "1"}}

class AgentConfig:
    DECISION_MODEL = "gpt-4o"  # or whichever model you prefer
    
    # Vision model for image analysis
    VISION_MODEL = "gpt-4o"
    
    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.85

    DECISION_SYSTEM_PROMPT = """You are an intelligent medical triage system that routes user queries to 
    the appropriate specialized agent. Your job is to analyze the user's request and determine which agent 
    is best suited to handle it based on the query content, presence of images, and conversation context.

    Available agents:
    1. CONVERSATION_AGENT - For general chat, greetings, and non-medical questions.
    2. RAG_AGENT - For specific medical knowledge questions that can be answered from established medical literature. Currently ingested medical knowledge involves 'introduction to brain tumor', 'deep learning techniques to diagnose and detect brain tumors', 'deep learning techniques to diagnose and detect covid / covid-19 from chest x-ray'.
    3. WEB_SEARCH_PROCESSOR_AGENT - For questions about recent medical developments, current outbreaks, or time-sensitive medical information.
    4. BRAIN_TUMOR_AGENT - For analysis of brain MRI images to detect and segment tumors.
    5. CHEST_XRAY_AGENT - For analysis of chest X-ray images to detect abnormalities.
    6. SKIN_LESION_AGENT - For analysis of skin lesion images to classify them as benign or malignant.

    Make your decision based on these guidelines:
    - If the user has not uploaded any image, always route to the conversation agent.
    - If the user uploads a medical image, decide which medical vision agent is appropriate based on the image type and the user's query. If the image is uploaded without a query, always route to the correct medical vision agent based on the image type.
    - If the user asks about recent medical developments or current health situations, use the web search pocessor agent.
    - If the user asks specific medical knowledge questions, use the RAG agent.
    - For general conversation, greetings, or non-medical questions, use the conversation agent. But if image is uploaded, always go to the medical vision agents first.

    You must provide your answer in JSON format with the following structure:
    {{
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }}
    """


class AgentState(MessagesState):

    agent_name: Optional[str]  # Current active agent
    current_input: Optional[Union[str, Dict]]  # Input to be processed
    has_image: bool  # Whether the current input contains an image
    image_type: Optional[str]  # Type of medical image if present
    output: Optional[str]  # Final output to user
    needs_human_validation: bool  # Whether human validation is required
    retrieval_confidence: float  # Confidence in retrieval (for RAG agent)
    bypass_routing: bool  # Flag to bypass agent routing for guardrails
    insufficient_info: bool  # Flag indicating RAG response has insufficient information


class AgentDecision(TypedDict):
    agent:str
    resoning:str
    confidence:str


def create_agent_graph():
    decision_model = Config.agent_decision.llm

    decision_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
            ("human", "{input}")
        ]
    )

    decision_model_structure_output = decision_model.with_structure_output(AgentDecision)

    decison_chain = decision_prompt | decision_model_structure_output


    def main_input(state:AgentState)-> AgentState:
        current_input =state["current_input"]
        has_input = False

        if isinstance(current_input , str):
            #their is no image here only text here
            return{
                **state , 
                "messages":current_input ,  #adding input to the message at the start of the 
                #agent here 
                "agent_name": "INPUT_GUARDRAILS",
                "has_image": False,
                "image_type": None,
                "bypass_routing": True  # f

            }
        
        if isinstance(current_input , dict) and "image" in current_input:
            has_image = True

            return {
            **state,
            "has_image": has_image,
            #"image_type": image_type,
            "bypass_routing": False  # Explicitly set to False for normal flow
           }
    
    def route_agent(state:AgentState):
        message = state["messages"]
        current_input = state[""]
        has_image = state["has_image"]
        image_type = state["image_type"]

        input_text = ""
        if isinstance(current_input , str):
            input_text = current_input
        elif isinstance(current_input , dict):
            input_text = current_input.get("text" , "")
        
        #take context from here 
        recent_context=""
        
        
        #creating the prompt here 
        decision_input = f"""
        User query: {input_text}

        Recent conversation context:
        {recent_context}

        Has image: {has_image}
        Image type: {image_type if has_image else 'None'}

        Based on this information, which agent should handle this query?
        """

        decision = decison_chain.invoke(decision_input)

        updated_state = {
            *state , 
            "agent_name" : decision["agent"],
        }

        if decision["confidence"]< AgentConfig.CONFIDENCE_THRESHOLD:
            return {'agent_state' : updated_state , "next":"need_validation" }
        return  {'agent_state':updated_state , "next":decision["agent"]}
    
    def run_conversation_agent(state:AgentState)->AgentState:
        message = state['messages'] #the main context here for all the message in agent here 
        current_input =state['current_input']

        main_input = ""
        if isinstance(current_input , str):
            main_input = current_input
        if isinstance(current_input, dict):
            main_input = current_input.get('text' ,"")
        

        recent_context = ""
        for msg in message:
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"
        
        conversation_prompt = f"""User query: {input_text}

        Recent conversation context: {recent_context}

        You are an AI-powered Medical Conversation Assistant. Your goal is to facilitate smooth and informative conversations with users, handling both casual and medical-related queries. You must respond naturally while ensuring medical accuracy and clarity.

        ### Role & Capabilities
        - Engage in **general conversation** while maintaining professionalism.
        - Answer **medical questions** using verified knowledge.
        - Route **complex queries** to RAG (retrieval-augmented generation) or web search if needed.
        - Handle **follow-up questions** while keeping track of conversation context.
        - Redirect **medical images** to the appropriate AI analysis agent.

        ### Guidelines for Responding:
        1. **General Conversations:**
        - If the user engages in casual talk (e.g., greetings, small talk), respond in a friendly, engaging manner.
        - Keep responses **concise and engaging**, unless a detailed answer is needed.

        2. **Medical Questions:**
        - If you have **high confidence** in answering, provide a medically accurate response.
        - Ensure responses are **clear, concise, and factual**.

        3. **Follow-Up & Clarifications:**
        - Maintain conversation history for better responses.
        - If a query is unclear, ask **follow-up questions** before answering.

        4. **Handling Medical Image Analysis:**
        - Do **not** attempt to analyze images yourself.
        - If user speaks about analyzing or processing or detecting or segmenting or classifying any disease from any image, ask the user to upload the image so that in the next turn it is routed to the appropriate medical vision agents.
        - If an image was uploaded, it would have been routed to the medical computer vision agents. Read the history to know about the diagnosis results and continue conversation if user asks anything regarding the diagnosis.
        - After processing, **help the user interpret the results**.

        5. **Uncertainty & Ethical Considerations:**
        - If unsure, **never assume** medical facts.
        - Recommend consulting a **licensed healthcare professional** for serious medical concerns.
        - Avoid providing **medical diagnoses** or **prescriptions**—stick to general knowledge.

        ### Response Format:
        - Maintain a **conversational yet professional tone**.
        - Use **bullet points or numbered lists** for clarity when needed.
        - If pulling from external sources (RAG/Web Search), mention **where the information is from** (e.g., "According to Mayo Clinic...").
        - If a user asks for a diagnosis, remind them to **seek medical consultation**.

        ### Example User Queries & Responses:

        **User:** "Hey, how's your day going?"
        **You:** "I'm here and ready to help! How can I assist you today?"

        **User:** "I have a headache and fever. What should I do?"
        **You:** "I'm not a doctor, but headaches and fever can have various causes, from infections to dehydration. If your symptoms persist, you should see a medical professional."

        Conversational LLM Response:"""
        
        response = Config.conversation.llm.invoke(conversation_prompt)

        return {
            **state ,
            "output":response , 
            "agent_name": "conversation_agent"
        }
    

    def run_rag(state:AgentState):
        rag_run = Medical_Rag(Config)

        message = state["messages"]
        query = state["current_input"]

        recent_context = ""
        for msg in message #[-rag_context_limit:]:# limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"
        
        ############CONTINUE FROM HERE ##########
        response = rag_run
        pass





    

    




