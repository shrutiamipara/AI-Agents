import math
import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent

# Load environment variables from .env file
load_dotenv()

@tool
def calculator(expression: str) -> str:
    """Evaluate a numeric Python expression. Input must be a pure numeric expression
    (no text, currency symbols, or '='). Use '**' for exponents.
    """
    try:
        result = eval(expression, {"math": math, "__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error calculating: {e}"


@tool
def lookup_user(user_id: int) -> str:
    """Simple mock lookup: returns user info for a known ID."""
    mock_db = {
        42: "Name: Alice Johnson, Role: Senior Engineer, Access: Admin",
        85: "Name: Bob Smith, Role: Data Analyst, Access: Standard",
    }
    return mock_db.get(int(user_id), f"User ID {user_id} not found in the database.")


tools = [calculator, lookup_user]

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Use tools when needed to answer mathematical questions or look up user data.",
)


if __name__ == "__main__":
    print("\nWelcome! The Professional AI Assistant is now online. You can seamlessly perform calculations or look up user data.")

    chat_history = []
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            if not user_input.strip():
                continue

            chat_history.append({"role": "user", "content": user_input})
            try:
                result = agent.invoke({"messages": chat_history})
                
                # Update chat_history to retain tool calls and complete message history
                chat_history = list(result["messages"])
                
                agent_response = chat_history[-1].content
                print(f"Agent:\n{agent_response}")
            except Exception as e:
                print(f"Agent Error: {e}")
                chat_history.pop()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")
