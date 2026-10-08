from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

def get_model():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=2048,
        max_retries=5,
    )

def text_chunking(text:str)->list[str]:
    """Function to chunk text into smaller segments for processing."""
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks=text_splitter.split_text(text)
    return chunks

def summarize_text(text:str)->str:
    """Summarization function that uses the Groq API to summarize text."""
    model=get_model()
    prompt=ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful assistant that summarizes the transcription. "
             "Return the key points clearly and concisely in no more than 150 words."),
            ("human",
             "{text}")
        ] 
    )
    chain=prompt|model|StrOutputParser()
    
    chunks=text_chunking(text)
    if not chunks:
        return "No content to summarize."

    summaries = [s.strip() for s in (chain.invoke({"text": chunk}) for chunk in chunks) if s.strip()]
    if not summaries:
        return "Summary could not be generated."

    if len(summaries) == 1:
        return summaries[0]
        
    final_input = "\n\n".join(summary[:800] for summary in summaries)
    
    prompt_final=ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are an expert summarizer. Combine the following summary points into "
             "one coherent final summary of no more than 250 words."),
            ("human",
             "{text}")
        ]
    )
    final_chain=prompt_final|model|StrOutputParser()
    final_summary=final_chain.invoke({"text": final_input}).strip()
    return final_summary

def generate_title(text:str)->str:
    model=get_model()
    
    prompt=ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant that generates a concise title for the transcription. Return ONLY the title in 5 words or less."),
        ("human",
         "{text}")
    ])
    
    chain=prompt|model|StrOutputParser()
    title=chain.invoke({"text": text[:2000]}).strip()
    return title
    
    


