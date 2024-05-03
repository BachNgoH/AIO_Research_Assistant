import streamlit as st 
import requests
from datetime import datetime



# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="localbot", page_icon="🧑‍💼", layout="wide")
    
    
def send_query(text, api_key):
    headers = {"Content-Type": "application/json"}
    data={"message": text, "api_key": api_key}
    resp = requests.post("http://localhost:8000/complete", json=data, headers=headers , stream=True)
    return resp

def run_app(username):
    st.sidebar.header("API Settings")
    api_key = st.sidebar.text_input("Enter Groq API Key", key="api_key", type="password")
    
    st.title("💬 Chatbot")
    st.caption("🚀 I'm a Local Bot")    
    # Function to append and display a new message
    def append_and_display_message(role, content):
        st.session_state.messages.append({"role": role, "content": content})
        st.chat_message(role).write(content)

    if "messages" not in st.session_state:
        # Load chat history from the database
        st.session_state["messages"] = []

        # Start with a greeting from the assistant if no history is found
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]
            
    # Initialize a unique chatID and session state for messages if not already present
    if "chatID" not in st.session_state:
        st.session_state["chatID"] = f"{username}-{str(datetime.now())}"

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    # Initialize the QA system using caching
    # translater = Translation(from_lang="en", to_lang='vi', mode='translate') 
    if query := st.chat_input():
        append_and_display_message("user", query)
        
        res = send_query(query, api_key)
            
        res.raise_for_status()
        
        answer = res.json()["completion"]

        # with st.chat_message("assistant"):
            # Create a placeholder for streaming messages
            # message_placeholder = st.empty()
            # full_response = ""

            # for chunk in res.iter_content(
            #     chunk_size=None, decode_unicode=True
            # ):
            #     full_response += chunk
            #     message_placeholder.markdown(full_response + "▌")

            # message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        st.chat_message("assistant").write(answer)
        
        # Save the chat history to the database

if __name__ == "__main__":
    username = "bachngo"
    run_app(username)