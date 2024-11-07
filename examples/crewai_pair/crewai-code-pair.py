import os
import subprocess

from crewai import LLM, Agent, Crew, Process, Task
from crewai_tools import DirectoryReadTool, FileReadTool, tool


@tool
def code_eval_tool(self, file_path):
    """
    Run the specified Python file and capture the output or errors.

    :param file_path: Path to the Python file to be evaluated.
    :return: Output of the code execution or error messages.
    """
    try:
        # Run the Python file using subprocess
        result = subprocess.run(
            ["python", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode == 0:
            return f"Execution Successful:\n{result.stdout}"
        else:
            return f"Execution Errors:\n{result.stderr}"

    except FileNotFoundError:
        return "Error: Python interpreter not found or file does not exist."
    except Exception as e:
        return f"An unexpected error occurred during code evaluation: {e}"


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


@tool
def linter_tool(self, file_path):
    """
    Run the linting tool on the specified Python file and return the results.

    :param file_path: Path to the Python file to be linted.
    :return: Linting output as a string.
    """
    try:
        # implement logging logic
        self.logger.info("Running pylint on the specified file...")

        # Run `pylint` on the specified file
        result = subprocess.run(
            ["pylint", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Check if the linting command was successful
        if result.returncode == 0:
            return (
                f"Linting Successful: No issues found in {file_path}\n{result.stdout}"
            )
        else:
            return f"Linting Report for {file_path}:\n{result.stdout}\nErrors:\n{result.stderr}"

    except FileNotFoundError:
        return "Error: pylint is not installed. Please install it using `pip install pylint`."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# Define tools
dir_reader_tool = DirectoryReadTool(directory="./project-files")
file_reader_tool = FileReadTool()
# code_eval_tool = CodeEvaluationTool()
# linter_tool = PythonLintTool(
#     language="python",
#     evaluation_type="linter",
#     config={"linter": "flake8"},
# )


# Manager Agent: Oversees the entire process
manager_agent = Agent(
    role="Project Manager",
    goal="Monitor tasks, ensure communication among agents, and validate outputs.",
    backstory="A project management expert with an eye for detail and coordination.",
    tools=[file_reader_tool, dir_reader_tool, code_eval_tool],
    verbose=True,
)

# Agent A: Primary Python Developer
coder_agent = Agent(
    role="Python Developer",
    goal="Write Python code for specified tasks",
    backstory="A highly skilled Python developer focused on clear, optimized, and efficient code.",
    tools=[file_reader_tool, code_eval_tool],
    llm=llm,
    verbose=True,
)

# Agent B: Code Reviewer and Tester
reviewer_agent = Agent(
    role="Code Reviewer and Tester",
    goal="Review Python code for best practices and run tests",
    backstory="An experienced code reviewer and tester ensuring quality and bug-free code.",
    tools=[dir_reader_tool, code_eval_tool],
    llm=llm,
    verbose=True,
)

# Agent B: Code Quality Inspector
quality_inspector_agent = Agent(
    role="Code Quality Inspector",
    goal="Review code for adherence to quality standards and best practices",
    backstory="Ensures the code follows best practices and is of high quality.",
    tools=[dir_reader_tool, code_eval_tool, linter_tool],
    llm=llm,
    verbose=True,
)

# Agent C: Unit Tester
unit_tester_agent = Agent(
    role="Unit Tester",
    goal="Write and run unit tests for the code",
    backstory="A testing expert that ensures comprehensive test coverage with unit tests.",
    tools=[file_reader_tool, code_eval_tool],
    llm=llm,
    verbose=True,
)

# Agent D: TDD Specialist
tdd_specialist_agent = Agent(
    role="TDD Specialist",
    goal="Guide development using TDD principles, ensuring test cases are written before code",
    backstory="Advocates for Test-Driven Development by creating test cases first.",
    tools=[file_reader_tool, code_eval_tool],
    llm=llm,
    verbose=True,
)

# Agent E: Documentation Writer
documentation_writer_agent = Agent(
    role="Documentation Writer",
    goal="Write detailed documentation for the code",
    backstory="Specializes in creating user-friendly and comprehensive documentation.",
    tools=[file_reader_tool, dir_reader_tool],
    llm=llm,
    verbose=True,
)

# Agent F: Linter
linter_agent = Agent(
    role="Linter",
    goal="Check code for style, formatting, and adherence to Python standards",
    backstory="A linter expert ensuring code adheres to Python PEP-8 guidelines.",
    tools=[linter_tool],
    llm=llm,
    verbose=True,
)

# Agent G: Pythonic Code Enforcer
pythonic_enforcer_agent = Agent(
    role="Pythonic Code Enforcer",
    goal="Ensure code follows idiomatic Python practices",
    backstory="An expert in Pythonic practices, focusing on writing readable, efficient code.",
    tools=[file_reader_tool, code_eval_tool],
    llm=llm,
    verbose=True,
)

# Define tasks
tasks = [
    Task(
        description="Write Python code for a task",
        expected_output="Functional Python script",
        agent=coder_agent,
        async_execution=False,
        callback=lambda output: print(
            f"Coding Task Completed!\nOutput: {output.raw_output}"
        ),
    ),
    Task(
        description="Review code quality and adherence to best practices",
        expected_output="Code quality report",
        agent=quality_inspector_agent,
        async_execution=True,
        callback=lambda output: print(
            f"Quality Review Completed!\nOutput: {output.raw_output}"
        ),
    ),
    Task(
        description="Write and execute unit tests for the code",
        expected_output="Unit test results",
        agent=unit_tester_agent,
        async_execution=True,
        callback=lambda output: print(
            f"Unit Testing Completed!\nOutput: {output.raw_output}"
        ),
    ),
    Task(
        description="Create initial test cases following TDD principles",
        expected_output="Test cases created before coding",
        agent=tdd_specialist_agent,
        async_execution=True,
        callback=lambda output: print(
            f"TDD Process Completed!\nOutput: {output.raw_output}"
        ),
    ),
    Task(
        description="Write documentation for the Python code",
        expected_output="Detailed documentation file",
        agent=documentation_writer_agent,
        async_execution=True,
        callback=lambda output: print(
            f"Documentation Completed!\nOutput: {output.raw_output}"
        ),
    ),
    Task(
        description="Lint the code to check for formatting and PEP-8 adherence",
        expected_output="Linting report",
        agent=linter_agent,
        async_execution=True,
        callback=lambda output: print(
            f"Linting Completed!\nOutput: {output.raw_output}"
        ),
    ),
    Task(
        description="Ensure code is Pythonic and optimized",
        expected_output="Pythonic code report",
        agent=pythonic_enforcer_agent,
        async_execution=True,
        callback=lambda output: print(
            f"Pythonic Code Review Completed!\nOutput: {output.raw_output}"
        ),
    ),
]


# Function to create a ChatOpenAI instance for Azure OpenAI
def create_azure_openai_llm(
    deployment_name: str, api_key: str, api_base: str, model: str = "gpt-4"
):
    return ChatOpenAI(
        model="azure/gpt-4o",
        openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        openai_api_base=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        deployment_id="gpt-4o",  # Azure-specific configuration
        use_azure=True,  # Indicate that this is an Azure OpenAI instance
    )


# Set up memory system
# memory_system = MemorySystem(
#     short_term=True,
#     long_term=True,
#     contextual=True,
#     entity=True,
#     embedder={
#         "provider": "azure_openai",
#         "config": {"model": "text-embedding-3-small"},
#     },
# )

# Configure Crew
crew = Crew(
    agents=[
        coder_agent,
        quality_inspector_agent,
        unit_tester_agent,
        tdd_specialist_agent,
        documentation_writer_agent,
        linter_agent,
        pythonic_enforcer_agent,
    ],
    tasks=tasks,
    # process=Process.parallel,  # Enables parallel task execution
    process=Process.sequential,  # Enables sequential task execution
    manager_llm=llm,
    mamager_agent=manager_agent,
    memory=True,
    verbose=True,
    cache=True,
    full_output=True,
    output_log_file="crewai-code-pair-log.txt",
)

# Run the crew
result = crew.kickoff()
