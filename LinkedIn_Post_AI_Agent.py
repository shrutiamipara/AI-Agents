from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=1
)

# Prompt Template
prompt = PromptTemplate(
    input_variables=["tone", "target", "audience"],
    template="""
You are an AI assistant optimized to write professional and engaging LinkedIn posts.

Generate a LinkedIn post based on the following:

Tone: {tone}
Target: {target}
Audience: {audience}

The post MUST be highly professional, concise, and suitable for LinkedIn. 
Please include relevant emojis to make the post engaging, while maintaining a professional tone. 
Include relevant hashtags at the end if the topic warrants it, and add a call to action if appropriate.
"""
)

# User Input
tone = input("Tone: ")
target = input("Target: ")
audience = input("Audience: ")

# Create Final Prompt
final_prompt = prompt.format(
    tone=tone,
    target=target,
    audience=audience
)

# Generate Response
response = llm.invoke(final_prompt)

# Print Output
print("\nGenerated LinkedIn Post:\n")
print(response.content)