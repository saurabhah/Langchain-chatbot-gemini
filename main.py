from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@tool
def search_csv():
    """Reads the dataframe from pandas."""
    df = pd.read_csv(os.getenv("CSV_PATH"))
    return df


agent = create_agent(
    model="google_genai:gemini-2.5-flash-lite",
    tools=[search_csv],
    system_prompt="You are a helpful assistant, who can use tools to answer questions about a CSV file. The CSV file contains data about employees, including their names, emails, and departments. Use the search_csv tool to read the dataframe and find the information needed to answer the user's query.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's is the email of Donald,OConnell? Please access the data from the csv file."}]}
)

print(result["messages"][-1].content_blocks)
