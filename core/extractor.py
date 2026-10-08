from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os
from dotenv import load_dotenv
load_dotenv()

def get_model():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=2048
    )

# Extract 
def build_chain(system_prompt:str):
    model=get_model()
    chain=RunnablePassthrough()|RunnableLambda(lambda x: {"text":x})|ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{text}")
    ])|model|StrOutputParser()
    
    return chain

def actionable(text:str)->str:
    """Function to extract actionable items from the text"""
    system_prompt=("You are an expert analyst. "
                   "Extract actionable items from the video transcription and provide:\n"
                   "1. A list of actionable items\n"
                   "2. Tasks to be done\n"
                   "3. Deadlines for each task\n"
                   "4. If there are no actionable items, please return 'No actionable items found.'")

    chain=build_chain(system_prompt)
    result=chain.invoke(text)
    return result

def decision(text:str, system_prompt:str|None=None)->str:
    if system_prompt is None:
        system_prompt=("You are an expert analyst. "
                       "Extract decisions from the video transcription and provide:\n"
                       "1. A list of decisions made\n"
                       "2. For each decision, provide the rationale behind it\n"
                       "3. If there are no decisions, please return 'No decisions found.'")
    chain=build_chain(system_prompt)
    result=chain.invoke(text)
    return result

def questions(text:str, system_prompt:str|None=None)->str:
    if system_prompt is None:
        system_prompt=("You are an expert analyst. "
                       "Extract questions asked or addressed in the video transcription and provide:\n"
                       "1. A list of questions asked or explored\n"
                       "2. For each question, provide the answer or key takeaway if available\n"
                       "3. If there are no questions, please return 'No questions found.'")
    chain=build_chain(system_prompt)
    result=chain.invoke(text)
    return result
