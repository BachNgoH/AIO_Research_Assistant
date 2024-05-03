import streamlit as st
import requests

# --- App Title and Configuration ---
st.set_page_config(page_title="Chat App", page_icon="💬")
st.title("Chat App")

# --- GROQ API Key Input ---
api_key = st.text_input("Enter your GROQ API Key:", type="password", key="api_key")

# --- User Input Area ---
user_input = st.text_input("You: ", key="input")

# --- Chat History ---
if 'chat_history' not in st.session_state:
   st.session_state['chat_history'] = []

# --- Function to call the API (Modified) ---
def get_bot_response(user_message):
    if api_key:  # Use the API key if provided
        api_endpoint = "http://"
        response = requests.post(api_endpoint, data={"message": user_message, "api_key": api_key})

        if response.status_code == 200:
            return response.json()["bot_response"] 
        else:
            return "Something went wrong with the API call."
    else:
        return "Please provide your GROQ API Key."

# --- Main App Logic ---
if user_input and st.session_state['input']:
  # Append user and bot messages to chat history
  st.session_state['chat_history'].append({"message": user_input, "is_user": True})
 
  bot_response = get_bot_response(user_input)  
  st.session_state['chat_history'].append({"message": bot_response, "is_user": False})

  # Clear the input 
  st.session_state['input'] = "" 

# --- Display Chat History ---
for chat_item in st.session_state['chat_history']:
  if chat_item['is_user']:
    st.write(f"**You:** {chat_item['message']}")
  else:
    st.write(f"**Bot:** {chat_item['message']}") 