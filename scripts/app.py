import gradio as gr
from chatbot_engine import chat_with_dear_zindagi

# This function only runs once when the user enters their name
def set_user_name(name):
    if name.strip():
        return gr.update(visible=True), name
    return gr.update(visible=False), ""

# This function handles the message submission
def respond(message, chat_history, user_name):
    if not user_name:
        return chat_history + [[message, "❗ Please enter your name before chatting."]], chat_history

    response = chat_with_dear_zindagi(message, user_name)
    chat_history.append((message, response))
    return chat_history, chat_history

with gr.Blocks() as demo:
    gr.Markdown("## Welcome to Dear Zindagi\nPlease enter your name to begin.")

    with gr.Row():
        name_input = gr.Textbox(label="Your Name", placeholder="Enter your name...")
        name_button = gr.Button("Start Chat")

    user_name_state = gr.State("")
    chatbot = gr.Chatbot(label="Dear Zindagi")
    msg = gr.Textbox(label="Message", placeholder="Type your message and press Enter...", visible=False)
    history_state = gr.State([])

    name_button.click(
        fn=set_user_name,
        inputs=name_input,
        outputs=[msg, user_name_state],
    )

    msg.submit(
        fn=respond,
        inputs=[msg, history_state, user_name_state],
        outputs=[chatbot, history_state],
    )

demo.launch()
