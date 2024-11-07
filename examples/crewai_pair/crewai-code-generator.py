import os

from dotenv import load_dotenv

load_dotenv("/.env")

from crewai import LLM, Agent, Crew, Task

# azure openAI
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

code_reviewer_agent = Agent(
    role="Senior Software Engineer",
    goal="Make sure the code written is optimized and maintainable",
    backstory="""You are a Senior software engineer who reviews the code written for a given task.
                               You should check the code for readability, maintainability, and performance.""",
    llm=llm,
    verbose=True,
)

code_reviewer_task = Task(
    description="""A software engineer has written this code for the given problem 
                            in the {language} programming language.' Review the code critically and 
                            make any changes to the code if necessary. 
                            'Problem: {problem}""",
    expected_output="Well formatted code after the review",
    agent=code_reviewer_agent,
)

# Create a agent: code_writer_agent
code_writer_agent = Agent(
    role="Software Engineer",
    goal="Write optimized code for a given task",
    backstory="""You are a software engineer who writes code for a given task.
                 The code should be optimized, and maintainable and 
                 include doc string, comments, etc.""",
    llm=llm,
    verbose=True,
)

# Create a agent: code_writer_task
code_writer_task = Task(
    description="Write the code to solve the given problem in the {language} programming language."
    "Problem: {problem}",
    expected_output="Well formatted code to solve the problem. Include type hinting",
    agent=code_writer_agent,
)

# Create a agent: code_reviewer_agent
code_reviewer_agent = Agent(
    role="Senior Software Engineer",
    goal="Make sure the code written is optimized and maintainable",
    backstory="""You are a Senior software engineer who reviews the code
                 written for a given task. You should check the code for
                 readability, maintainability, and performance.""",
    llm=llm,
    verbose=True,
)

# Create a task: code_reviewer_task
code_reviewer_task = Task(
    description="""A software engineer has written this code for the given problem
                   in the {language} programming language.' Review the code critically and
                   make any changes to the code if necessary.
                   '\nProblem: {problem}""",
    expected_output="Well formatted code after the review",
    agent=code_reviewer_agent,
)

code_requirements_agent = Agent(
    role="Data Scientist",
    goal="provide are all things that should be required in the code to solve the given problem.",
    backstory="""You are a Data Scientist who decides what are all things required
                 in the code to solve a given problem/task. The code will be written based on
                 the requirements provided by you.""",
    llm=llm,
    verbose=True,
)

code_requirement_task = Task(
    description="Write the requirements for the given problem step-by-step."
    "Problem: {problem}",
    expected_output="Well formatted text which specifies what is required to solve the problem.",
    agent=code_requirements_agent,
    human_input=True,
)

# Create a crew with the agents and tasks defined above
# and kickoff the process
crew = Crew(
    agents=[code_writer_agent, code_reviewer_agent],
    tasks=[code_writer_task, code_reviewer_task],
    llm=llm,
    verbose=True,
)

# Kickoff the process
result = crew.kickoff(
    inputs={"problem": "create a game of tic-tac-toe", "language": "Python"}
)

print(result.dict())
print(result.dict()["token_usage"])
print(result.raw)
