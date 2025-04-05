import gradio as gr
from chatbot_engine import chat_with_dear_zindagi

# Session state
chat_history = []

# Ask user's name first
def capture_name(name):
    if not name.strip():
        return (
            gr.update(placeholder="Please enter your name."),  # name_input
            gr.update(visible=True),                           # start_button
            gr.update(visible=False),                          # msg
            gr.update(visible=False),                          # chatbot
            gr.update(visible=False),                          # send
            None,                                              # user_name_state (no change)
            []                                                 # chat_history (no change)
        )
    
    # greeting = f"Hi {name}, I'm here to listen. How are you feeling today?"
    greeting = chat_with_dear_zindagi(
        f"Please greet the user named {name} and let them know you're here to listen and support them.",
        user_name=name
    )
    chat_history.clear()
    chat_history.append({"role": "assistant", "content": greeting})
    
    return (
        gr.update(visible=False),             # name_input
        gr.update(visible=False),             # start_button
        gr.update(visible=True),              # msg
        gr.update(visible=True),              # chatbot
        gr.update(visible=True),              # send
        name,                                 # user_name_state properly updated
        chat_history                          # chat_history updated with greeting
    )

# Chat response logic
def respond(message, user_name_state):
    if not message.strip():
        return gr.update(), chat_history

    user_name = user_name_state.value if hasattr(user_name_state, "value") else user_name_state
    chat_history.append({"role": "user", "content": message})
    reply = chat_with_dear_zindagi(message, user_name=user_name)
    chat_history.append({"role": "assistant", "content": reply})
    return "", chat_history


with gr.Blocks(title="Dear Zindagi") as demo:
    gr.Markdown("## Welcome to Dear Zindagi — Your Mental Health Companion")
    gr.Markdown("### A mental health chatbot powered by AI")

    user_name_state = gr.State()  # needs to be inside the demo block

    name_input = gr.Textbox(
        placeholder="Enter your name to begin",
        label="Your Name",
        interactive=True,
        visible=True
    )
    start_button = gr.Button("Start Chat", visible=True)

    chatbot = gr.Chatbot(label="Dear Zindagi", visible=False, height=400, type="messages")
    msg = gr.Textbox(placeholder="Type your message...", visible=False, label="Your Message")
    send = gr.Button("Send", visible=False)

    # Hooking up name input
    start_button.click(
        capture_name,
        inputs=[name_input],
        outputs=[name_input, start_button, msg, chatbot, send, user_name_state, chatbot]
    )
    name_input.submit(
        capture_name,
        inputs=[name_input],
        outputs=[name_input, start_button, msg, chatbot, send, user_name_state, chatbot]
    )

    # Hooking up chat interaction
    send.click(
        respond,
        inputs=[msg, user_name_state],
        outputs=[msg, chatbot]
    )
    msg.submit(
        respond,
        inputs=[msg, user_name_state],
        outputs=[msg, chatbot]
    )

demo.launch()
