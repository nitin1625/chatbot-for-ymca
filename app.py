import streamlit as st
from back import askPDFPost

# Streamlit app
st.title('CampusAI : Chatbot For YMCA')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask Something...")

if query:
    with st.chat_message('user'):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    response = askPDFPost(query=query)
    with st.chat_message('assistant'):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
