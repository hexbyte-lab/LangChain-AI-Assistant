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


def chat_with_tesla(message, history):
    """
    Chat function for Gradio interface
    Args:
        message: Current user message (string or dict)
        history: List of previous messages
    Returns:
        Response from Tesla
    """
    # Extract text if message is a dict
    if isinstance(message, dict):
        user_text = message.get("text", message.get("content", ""))
    else:
        user_text = message

    # Build conversation history
    messages = [{"role": "system", "content": TESLA_SYSTEM_PROMPT}]

    # Add history - handle both tuple and dict formats
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append({"role": msg["role"], "content": msg["content"]})
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            # Old tuple format: [user_msg, bot_msg]
            messages.append({"role": "user", "content": msg[0]})
            messages.append({"role": "assistant", "content": msg[1]})

    # Add latest user message
    messages.append({"role": "user", "content": user_text})

    response = llm.invoke(messages)
    return response.content.strip()


demo = gr.ChatInterface(
    fn=chat_with_tesla,
    # Remove type="messages" - not needed in Gradio 5.x
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
    demo.launch(share=True)
