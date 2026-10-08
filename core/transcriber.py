import os
from dotenv import load_dotenv
import whisper

load_dotenv()

WHISPER_MODEL=os.getenv("WHISPER_MODEL","small")

model_eng=None

def load_model():
    global model_eng
    if model_eng is None:
        print(f"Loading Whisper model ({WHISPER_MODEL})...")
        model_eng=whisper.load_model(WHISPER_MODEL)
    else:
        print("Whisper model already loaded.")
        
    return model_eng

def transcribe_audio(audio_path:str, language:str="en")->str:
    """Transcribe audio using OpenAI's Whisper model."""

    model=load_model()
    kwargs = {"language": language} if language else {}
    result=model.transcribe(audio_path, **kwargs)
    return result["text"]

def transcribe_eng_audio(audio_path:str)->str:
    """Transcribe audio for the English language using OpenAI's Whisper model."""
    return transcribe_audio(audio_path, language="en")


def transcribe_chunks(chunks:list, language:str="en")->str:
    """Transcribe a list of audio chunks and concatenate the results"""
    full_transcription=""
    
    for i,chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}: {chunk}")
        
        transcription=transcribe_audio(chunk, language=language)
        full_transcription+=transcription+" "
    
    return full_transcription.strip()
    
    

        
