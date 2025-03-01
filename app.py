import streamlit as st

from augmentation import augmentation
from generation import generation
from retrieval import get_similar_texts
from retrieval import load_chunks_with_embeddings
from dotenv import load_dotenv

load_dotenv()

def chat(prompt):
    st.session_state.disabled = True

    # Add user message to chat history
    st.session_state.messages.append(("human", prompt))

    # Display assistant response in chat message container
    with st.chat_message("ai"):
      # Get complete chat history, including latest question as last message
      history = "\n".join(
        [f"{role}: {msg}" for role, msg in st.session_state.messages]
      )

      query = f"{history}\nAI:"

      embeddings, pages_and_chunks = load_chunks_with_embeddings()

      relevant_chunks = get_similar_texts(query, embeddings, pages_and_chunks)
      prompts = augmentation(query, relevant_chunks)
      output = generation(prompts)

      print("output from the model is: ")
      print(output)

      placeholder = st.empty()

      # write response without "▌" to indicate completed message.
      with placeholder:
        st.markdown(output)

    # Log AI response to chat history
    st.session_state.messages.append(("ai", output))
    # Unblock chat input
    st.session_state.disabled = False

    st.rerun()

if "disabled" not in st.session_state:
    # `disable` flag to prevent user from sending messages whilst the AI is responding
    st.session_state.disabled = False

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="RAG for Military Usecases")
st.title("Welcome to the RAG demo for Military Usecase")

with st.chat_message("ai"):
    st.markdown(
        f"Hi! I'm your AI assistant."
    )

# Display chat messages from history on app rerun
for role, message in st.session_state.messages:
    if role == "system":
        continue
    with st.chat_message(role):
        st.markdown(message)

current_chat_message = st.container()
prompt = st.chat_input("Ask your question here...", disabled=st.session_state.disabled)

if prompt:
    chat(prompt)

