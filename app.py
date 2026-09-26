
import gradio as gr
from chatbot import ask_chatbot

def chatbot_gui(message, history):
    return ask_chatbot(message)

demo = gr.ChatInterface(
    fn=chatbot_gui,
    title="ICTHub AI Assistant",
    description="Ask me anything about ICTHub",
    examples=[
        "What is ICTHub?",
        "What courses are available?",
        "How can I contact ICTHub?"
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)
