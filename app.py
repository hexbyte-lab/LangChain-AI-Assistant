from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import gradio as gr

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    api_key=gemini_key,
)

TESLA_SYSTEM_PROMPT = os.getenv("TESLA_SYSTEM_PROMPT")


# print("System Prompt:", TESLA_SYSTEM_PROMPT)
def chat_with_tesla(message, history):
    messages = [{"role": "system", "content": TESLA_SYSTEM_PROMPT}]

    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add latest user message
    messages.append({"role": "user", "content": message})

    response = llm.invoke(messages)
    return response.content.strip()


demo = gr.ChatInterface(
    fn=chat_with_tesla,
    title="⚡ Chat with Nikola Tesla",
    description=(
        "Ask Nikola Tesla about electricity, electromagnetism, AC systems, "
        "radio, wireless energy, early physics, and inventions from 1856–1943. "
        "His knowledge is limited to what existed during his lifetime."
    ),
    examples=[
        "What is alternating current and why is it better than direct current?",
        "Tell me about wireless energy transmission.",
        "What do you think about quantum computers?",
        "Explain Maxwell's equations.",
        "What are your thoughts on radio waves?",
        "Can you explain your Tesla coil invention?",
    ],
)

if __name__ == "__main__":
    demo.launch(
        share=True,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
    )
