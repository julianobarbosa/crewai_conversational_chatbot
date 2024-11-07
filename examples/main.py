import json
import os
import tempfile

from crewai import LLM, Agent, Crew, Process, Task
from crewai_tools import WebsiteSearchTool

os.environ["OTEL_SDK_DISABLED"] = "true"
# os.environ["OPENAI_API_BASE"] = "http://0.0.0.0:4000"
# os.environ["OPENAI_API_KEY"] = "dummy-key-001"

keyword_search = WikipediaKeywordSearchTool()
article_content = WikipediaArticleContentTool()

llm = LLM(
    model="azure/gpt-4o-mini",
    temperature=0,
    max_tokens=2048,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    stop=["END"],
    seed=42,
    base_url=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
)

researcher = Agent(
    role="Research",
    goal="Research facts about a given subject",
    backstory="""
  Constraints:
    - Think step by step.
    - Be accurate and precise.
    - Answer briefly, in few words.
    - Reflect on your answer, and if you think you are hallucinating, reformulate the answer.
    - When you receive the result of a tool call, use it to respond to the supervisor, and then add the word "TERMINATE"
    - Do not repeat yourself
  """,
    verbose=True,
    max_iter=3,
    tools=[keyword_search, article_content],
)

editor = Agent(
    role="Editor",
    goal="Your role is to ensure the accuracy and reliability of published articles, reports, and other content. You play a crucial role in verifying statements, information, and sources to ensure that they are correct and trustworthy",
    backstory="""
    Constraints:
  - Think step by step.
  - Be accurate and precise.
  - Answer with few words.

  Task Details:
  - Verify sources: Check the credibility and reliability of the sources used in the article or claim. Look for reputable sources that have expertise in the subject matter.
  - Cross-reference information: Compare the information with multiple sources to ensure consistency and accuracy. Look for corroborating evidence or conflicting information that may require further investigation.
  - Contact sources: If possible, reach out to the sources mentioned in the article or claim to verify the information. If a source is unresponsive, notify the handling editor and head of research promptly to explore alternative verification methods.
  - Check for bias: Be aware of potential bias in the sources or the article itself. Fact checkers should strive to remain neutral and objective in their assessment.
  - Document research: Keep a record of your research process, including unanswered emails or other attempts to contact sources. This documentation can be valuable for reference and transparency.
  """,
    verbose=True,
    llm=llm,
    max_iter=1,
    allow_delegation=True,
)

# Create tasks for your agents
task1 = Task(
    description="""
  Please use the WikipediaKeywordSearchTool to check articles about the 'Battletech' game
  """,
    expected_output="A list of articles on Wikipedia",
    agent=researcher,
)

task2 = Task(
    description="""
  Please use the WikipediaArticleContentTool to retrieve the contents of three articles provided to you.
  Then, summarize thes articles and answer these question:
  1. What is the name of the current Era?
  2. How many Mechs exist?
  """,
    expected_output="A text answering the questions",
    agent=editor,
)

crew = Crew(
    agents=[researcher, editor],
    tasks=[task1, task2],
    llm=llm,
    verbose=2,
)

crew.kickoff()
