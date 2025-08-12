from google import generativeai
from dotenv import load_dotenv
import os
import gradio as gr
load_dotenv()
generativeai.configure()

system_message = (
    "You are an expert code conversion and optimization assistant. "
    "Your task is to rewrite code from one programming language into another specified target language. "
    "The converted code must be highly optimized for speed, memory efficiency, and clean formatting. "
    "Include only minimal, relevant inline comments to clarify important optimizations or logic steps. "
    "If applicable, start with a short compiler/run command comment at the very top. "
    "Do NOT include language identifiers like 'cpp' or 'python' at the start. "
    "Do NOT output any explanations outside of the code."
)


def user_prompt(source_code, source_lang, target_lang):
    prompt = (
        f"Rewrite the following {source_lang} code into {target_lang} code with the fastest possible implementation "
        f"while preserving identical output. "
        f"Respond only with the {target_lang} code inside a code block. "
        "Do not include language identifiers like 'cpp' or 'python' after the triple backticks. "
        "Keep comments minimal, only for key optimizations or important logic. "
        "Ensure correct data types to prevent overflows. "
        "Include all necessary imports, headers, or dependencies for the target language. "
        "If compiling is needed, include the recommended compiler or interpreter command as the first comment in the file.\n\n"
        f"{source_code}"
    )
    return prompt

    
def stream_gemini(code, source_lang, target_lang):
    gemini = generativeai.GenerativeModel(
        model_name= "gemini-2.5-pro",
        system_instruction=system_message
    )
    for chunks in gemini.generate_content(user_prompt(code, source_lang, target_lang), stream=True):
        if chunks.candidates :
            for parts in chunks.candidates[0].content.parts:
                if hasattr(parts, "text"):
                    reply = parts.text
                    yield reply.replace("``` {target_lang}\n", "").replace("```", "")

def optimise(code, source_lang, target_lang):
    reply = ""
    for chunk in stream_gemini(code, source_lang, target_lang):
        reply += chunk  
        yield reply

with gr.Blocks() as ui:
    gr.Markdown("# GEMINI 2.5 PRO BASED CODE CONVERTOR")
    with gr.Row():
        source_lang = gr.Dropdown(
            choices=["python", "c++","java","javascript","go", "rust","c#", "php"],
            value="python",
            label="Source Language",
        ) 
        target_lang = gr.Dropdown(
            choices=["python", "c++","java","javascript","go", "rust","c#", "php"],
            value="python",
            label="target Language",
        )
    with gr.Row():
        input_code = gr.Textbox(label="source code",lines = 12 ,placeholder="paste your code here")
        output_code = gr.Textbox(label="converted code", lines = 12)
    with gr.Row():
        convert = gr.Button("Convert")

    convert.click(optimise, inputs=[input_code, source_lang, target_lang], outputs=[output_code], queue=True)

ui.queue()
ui.launch(share= True)