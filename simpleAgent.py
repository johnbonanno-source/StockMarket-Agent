from collections import deque
import streamlit as st

import typing
import os
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemini-3-flash-preview",  # Model identifier.
    base_url="https://ollama.com",  # Ollama-compatible API endpoint.
    api_key=os.environ.get("OLLAMA_API_KEY"),  # Read API key from the environment.
)

def main_loop(system_prompt:str, history:deque):

    print("\n===AGENT===")
    while True:
        
        
        user_input = input("\nWhat is your question: ")

        if user_input.lower() in ['exit', 'quit']:
            break

        if user_input == 'agent_clear_context':
            history.clear()
        history.append(("human", user_input))
        # prompt_messages = [("system", system_prompt)] + list(history)
        resp = llm.invoke([("system", system_prompt)] + list(history))
        history.append(("assistant", resp.content))

        print("\n===ANSWER ===")
        print(resp.content)

if __name__ == "__main__":
    system_prompt = "You are a concise assistant. You follow up every response with a follow on question, on a new line, similar to the following, but create variation ex: Would you like to know more about a specific topic, or anything else? I will be here"
    d = deque()
    main_loop(system_prompt,d)
