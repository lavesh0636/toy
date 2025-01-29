import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.googlesearch import GoogleSearch
import groq
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the agent
@st.cache_resource
def get_agent():
    return Agent(
        model=Groq(
            id="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")),
        tools=[GoogleSearch()],
        description="Expert Ayurvedic consultant specializing in Gupt Rog and sexual health",
        instructions=[
            "Respond exclusively to Gupt Rog and Ayurvedic sexual health queries.",
            "Provide structured, markdown-formatted answers with practical solutions.",
            "Include detailed explanations, treatment options, and preventive measures.",
            "Maintain a professional and empathetic tone."
        ],
        markdown=True
    )

def handle_query(query: str) -> str:
    agent = get_agent()
    max_retries = 2
    
    for attempt in range(max_retries + 1):
        try:
            response = agent.run(query)
            # Extract just the clean response content
            if response and response.content:
                return response.content.strip()  # Clean up whitespace
            return "No response generated"
        except groq.InternalServerError as e:
            if "503" in str(e) and attempt < max_retries:
                time.sleep(5 * (attempt + 1))
                continue
            return "⚠️ Service temporarily unavailable. Please try again later."
        except Exception as e:
            return f"❌ Error processing request: {str(e)}"
    return "⚠️ Maximum retries exceeded. Please try again later."

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Gupt Rog Ayurvedic Expert",
        page_icon="🌿",
        layout="centered"
    )
    
    # Initialize session state for chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Header
    st.title("🌿 Gupt Rog Ayurvedic Consultant")
    st.markdown("""
    Welcome to the Ayurvedic Expert System for Gupt Rog (Hidden Diseases) and Sexual Health.
    Ask your questions in English or Hindi about:
    - Sexual health issues
    - Ayurvedic treatments
    - Preventive measures
    - Herbal remedies
    """)

    # Conversation history display
    for qa in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(qa["question"])
        with st.chat_message("assistant"):
            st.markdown(qa["answer"])

    # Query input field
    query = st.chat_input("Ask your question about Gupt Rog...")
    
    if query:
        # Basic relevance check for keywords related to Gupt Rog
        if not any(keyword in query.lower() for keyword in ["gupt rog", "sexual", "ayurved", "health","hidden diseases", "treatment"]):
            with st.chat_message("assistant"):
                st.markdown("I specialize in Gupt Rog and Ayurvedic sexual health. Please ask related questions.")
            return

        # Process the user's query with enhanced context handling
        with st.spinner("Consulting Ayurvedic texts..."):
            response = handle_query(query)
        
        # Store the question and response in history
        st.session_state.history.append({
            "question": query,
            "answer": response
        })
        
        # Display new messages in chat format
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            st.markdown(response)

    # Sidebar controls for additional settings
    with st.sidebar:
        st.header("Settings")
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
        st.markdown("---")
        st.markdown("**About:**\nAyurvedic AI Expert powered by Groq & Phi")

if __name__ == "__main__":
    main()
