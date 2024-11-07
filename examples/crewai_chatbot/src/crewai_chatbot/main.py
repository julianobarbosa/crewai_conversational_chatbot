#!/usr/bin/env python
import os
import sys

from mem0 import Memory

from crewai_chatbot.crew import CrewaiChatbotCrew

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "chatbot_memory",
            "path": "data/chrome_db",
        },
    },
}

memory = Memory.from_config(config_dict=config)


def run():
    """
    Run the crew.
    """
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! It was nice talking to you.")
            break

        # Add user input to memory
        # memory.add(f"User: {user_input}", user_id="user")

        # Retrieve relevant information from vector store
        # relevant_info = memory.search(query=user_input, limit=3)
        # context = "\n".join(message["memory"] for message in relevant_info)

        inputs = {
            "user_message": f"{user_input}",
            "context": f"{context}",
        }

        response = CrewaiChatbotCrew().crew().kickoff(inputs=inputs)

        # Add chatbot response to memory
        memory.add(f"Assistant: {response}", user_id="assistant")
        print(f"Assistant: {response}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {"topic": "AI LLMs"}
    try:
        CrewaiChatbotCrew().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        CrewaiChatbotCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {"topic": "AI LLMs"}
    try:
        CrewaiChatbotCrew().crew().test(
            n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
