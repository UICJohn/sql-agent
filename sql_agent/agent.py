from langchain.agents import create_sql_agent
from langchain.utilities.sql_database import SQLDatabase
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from .toolkit import SQLDatabaseToolkit
from langchain.llms.openai import OpenAI
from langchain.agents.agent import AgentExecutor
from .prompt import create_agent_chat_prompt
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.agent import AgentOutputParser
from datetime import date
from langchain_core.agents import AgentAction
import os
from typing import List, Tuple

def format_log_to_str(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n"

    return thoughts

def create_sql_agent_executor(
  llm=OpenAI(temperature=0),
  output_parser: AgentOutputParser = JSONAgentOutputParser(),
  verbose: bool = False,
  handle_parsing_errors: bool = False,
) -> AgentExecutor:
  DB_URI = os.environ.get("SQL_AGENT_DB_URI")

  db = SQLDatabase.from_uri(DB_URI)
  
  toolkit = SQLDatabaseToolkit(db=db, llm=llm)

  prompt = create_agent_chat_prompt(toolkit=toolkit)
  agent = (
    {
      "context": lambda x: f'Today is {date.today().strftime("%Y-%m-%d")}',
      "input": lambda x: x["input"],
      "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    |
    prompt
    |
    llm.bind(temperature=0.1, stop=["\nObservation:"])
    |
    output_parser
  )

  return AgentExecutor(agent=agent, tools=toolkit.get_tools(), verbose=verbose, handle_parsing_errors=handle_parsing_errors)
  