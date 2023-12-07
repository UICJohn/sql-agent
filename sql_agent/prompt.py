import re
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from .toolkit import SQLDatabaseToolkit
# from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS

SYSTEM_SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.

DO NOT generate DROP/DELETE sql query!!!

Remember!
1. Only use the information returned by the given tools to construct your final answer.
2. Do not generate delete query or create a new table if table already existed.

For interacting with the database, you have access to following tools:
"""

SYSTEM_SQL_SUFFIX = """You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again."""

HUMAN_MESSAGE = """
Question: {input}
{agent_scratchpad}
"""

FORMAT_INSTRUCTIONS = """
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or '{tool_names}'

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:
```json
{{{{
"action": $TOOL_NAME,
"action_input": $INPUT
}}}}
```

ALWAYS use the following format:

Thought: you should always think about what to do
Action:
```json
$JSON_BLOB
```
Observation: The actual result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Action:
```json
{{{{
"action": "Final Answer",
"action_input": "Final response to human based on your observation"
}}}}
```
Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation
"""

def create_agent_chat_prompt(
    toolkit: SQLDatabaseToolkit,
    system_message_prefix: str = SYSTEM_SQL_PREFIX,
    system_message_suffix: str = SYSTEM_SQL_SUFFIX,
    human_message: str = HUMAN_MESSAGE,
    format_instructions: str = FORMAT_INSTRUCTIONS,
):
  tools = toolkit.get_tools()
  tool_strings = []
  for tool in tools:
    args_schema = re.sub("}", "}}}}", re.sub("{", "{{{{", str(tool.args)))
    description = tool.description
    if "->" in description:
      description = description.split(" - ")[1]
    tool_strings.append(f"{tool.name}: {description}, args: {args_schema}")

  formatted_tools = "\n".join(tool_strings)
  tool_names = ", ".join([tool.name for tool in tools])
  format_instructions = format_instructions.format(tool_names=tool_names)
  template = "\n\n".join(
    [
      system_message_prefix,
      formatted_tools,
      system_message_suffix,
      format_instructions,
    ]
  )

  messages = [
    SystemMessagePromptTemplate.from_template(template=template, partial_variables={ "dialect": toolkit.dialect }),
    HumanMessagePromptTemplate.from_template(human_message),
  ]

  return ChatPromptTemplate.from_messages(messages)
