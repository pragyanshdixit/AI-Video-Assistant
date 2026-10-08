import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from utils.audio_preprocessing import process_input
from core.transcriber import transcribe_chunks
from core.summization import summarize_text, generate_title 
from core.extractor import actionable, decision, questions

link="https://youtu.be/T-D1OfcDW1M?si=CK2ISohhv6o8g8FZ"

print("Processing input...")
chunks=process_input(link)

transcription=transcribe_chunks(chunks)
print("Transcription completed.")

print("Generating summary...")
summary=summarize_text(transcription)
title=generate_title(transcription)
print(f"Title: {title}")
print(f"Summary: {summary[:500]}...")  # Print first 500 characters of the summary

print("Extracting actionable items...")
actionable_items=actionable(transcription)
print(f"Actionable Items: {actionable_items}")

print("Extracting decisions...")
decisions=decision(transcription)
print(f"Decisions: {decisions}")

print("Extracting questions...")
questions_list=questions(transcription)
print(f"Questions: {questions_list}")