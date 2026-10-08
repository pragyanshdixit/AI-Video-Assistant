import sys
import os
from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from utils.audio_preprocessing import process_input
from core.transcriber import transcribe_chunks
from core.summization import summarize_text, generate_title
from core.extractor import actionable, decision, questions
from core.rag_engine import build_rag_chain, ask_question



def load_pipeline(link:str):
    
    chunks=process_input(link)

    transcription=transcribe_chunks(chunks)
    
    # Automatically clean up audio and chunk files once transcribed
    if hasattr(chunks, "cleanup"):
        chunks.cleanup()
        print("Audio artifacts cleaned up to free disk space.")
    
    summary=summarize_text(transcription)
    title=generate_title(transcription)

    actionable_items=actionable(transcription)

    decisions=decision(transcription)

    questions_list=questions(transcription)

    rag_chain=build_rag_chain(transcription)
    
    
    return {
        "title": title,
        "summary": summary,
        "actionable_items": actionable_items,
        "decisions": decisions,
        "questions": questions_list,
        "rag_chain": rag_chain
        
    }
    
if __name__=="__main__":
    link=input("Enter the YouTube link or local audio path: ")
    pipeline_output=load_pipeline(link)
    
    print("\n" + "="*50)
    print(f"Title: {pipeline_output['title']}")
    print("="*50 + "\n")
    
    print(f"Summary:\n{pipeline_output['summary']}\n\n")
    print(f"Actionable Items:\n{pipeline_output['actionable_items']}\n\n")
    print(f"Decisions:\n{pipeline_output['decisions']}\n\n")
    print(f"Questions:\n{pipeline_output['questions']}\n\n")
    
    print("="*50)
    print("You can now ask questions based on the video content. Type 'exit' to quit.")
    print("="*50)
    while True:
        user_question = input("\nEnter your question: ").strip()
        if not user_question:
            continue
        if user_question.lower() == 'exit':
            break
        answer = ask_question(pipeline_output['rag_chain'], user_question)
        print(f"\nAnswer: {answer}")
    
        