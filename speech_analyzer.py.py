import os
import torch
import gradio as gr
from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Initialize the Endpoint LLM with your token passed directly as a string
llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    temperature=0.1,
    max_new_tokens=800,
    huggingfacehub_api_token="hf_your_token_here"  # Replace with your actual token
)

# Wrap it with ChatHuggingFace for conversational models
llm = ChatHuggingFace(llm=llm_endpoint)

#######------------- Prompt Template & LCEL Chain -------------####
pt = ChatPromptTemplate.from_messages([
    ("system", "List the key points with details from the context."),
    ("human", "The context : {context}")
])

prompt_to_LLAMA2 = pt | llm

#######------------- Speech2text -------------####
def transcript_audio(audio_file):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
        ignore_warning=True
    )
    transcript_txt = pipe(audio_file, batch_size=8)["text"]
    result = prompt_to_LLAMA2.invoke({"context": transcript_txt})
    return result.content

#######------------- Gradio -------------####
audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=output_text,
    title="Audio Transcription App",
    description="Upload the audio file"
)

if __name__ == "__main__":
    iface.launch()
