import os

from crewai import LLM, Agent, Crew, Task
from crewai_tools import DirectoryReadTool, FileReadTool

file_read_tool = FileReadTool("EDA.py")

# azure openAI
llm = LLM(
    model="azure/gpt-4o",
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

code_evaluator_agent = Agent(
    role="Data Science Evaluator",
    goal="Evaluate the given code file based on the requirements provided for a given problem",
    backstory="""You are a Data Science evaluator who reviews and evaluates the code.
                 You should check the code based on the requirements given to you""",
    llm=llm,
    verbose=True,
)

code_evaluator_task = Task(
    description="""A code file is given to you.
                   Evaluate the file based on the requirements given as the context.
                   Provide the only review and evaluation of the code as the output, not the code.
                """,
    expected_output="Detailed evaluation results of the code file based on the requirements."
    "Review the code file for each point of the requirements given to you"
    "Provide evaluation results as text",
    tools=[file_read_tool],
    agent=code_evaluator_agent,
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

crew = Crew(
    agents=[code_requirements_agent, code_evaluator_agent],
    tasks=[code_requirement_task, code_evaluator_task],
    verbose=True,
)

problem = """
Perform EDA on the NYC taxi trip duration dataset.
Here is the description of all the variables/features available in the dataset which will help you to perform EDA:

    id - a unique identifier for each trip
    vendor_id - a code indicating the provider associated with the trip record
    pickup_datetime - date and time when the meter was engaged
    dropoff_datetime - date and time when the meter was disengaged
    passenger_count - the number of passengers in the vehicle (driver entered value)
    pickup_longitude - the longitude where the meter was engaged
    pickup_latitude - the latitude where the meter was engaged
    dropoff_longitude - the longitude where the meter was disengaged
    dropoff_latitude - the latitude where the meter was disengaged
    store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server (Y=store and forward; N=not a store and forward trip)
    trip_duration - (target) duration of the trip in seconds

"""

result = crew.kickoff(inputs={"problem": problem})
