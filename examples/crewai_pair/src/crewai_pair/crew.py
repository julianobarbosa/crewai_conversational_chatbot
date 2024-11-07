import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Uncomment the following line to use an example of a custom tool
# from crewai_pair.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool


@CrewBase
class CrewaiPairCrew:
    """CrewaiPair crew"""

    def __init__(self):
        self.llm = LLM(
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

    @agent
    def coder_python(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_python"],
            # tools=[MyCustomTool()], # Example of custom tool, loaded on the beginning of file
            llm=self.llm,
            verbose=True,
        )

    @agent
    def coder_qa(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_qa"],
            # tools=[MyCustomTool()], # Example of custom tool, loaded on the beginning of file
            llm=self.llm,
            verbose=True,
        )

    @agent
    def coder_debugger(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_debugger"], llm=self.llm, verbose=True
        )

    @agent
    def coder_tester(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_tester"], llm=self.llm, verbose=True
        )

    @agent
    def coder_optimizer(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_optimizer"], llm=self.llm, verbose=True
        )

    @agent
    def coder_scalability(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_scalability"], llm=self.llm, verbose=True
        )

    @agent
    def coder_doc(self) -> Agent:
        return Agent(config=self.agents_config["coder_doc"], llm=self.llm, verbose=True)

    @agent
    def coder_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_reviewer"], llm=self.llm, verbose=True
        )

    @agent
    def coder_devops(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_devops"], llm=self.llm, verbose=True
        )

    @agent
    def coder_frontend(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_frontend"], llm=self.llm, verbose=True
        )

    @agent
    def coder_backend(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_backend"], llm=self.llm, verbose=True
        )

    @agent
    def coder_data(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_data"], llm=self.llm, verbose=True
        )

    @agent
    def coder_mobile(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_mobile"], llm=self.llm, verbose=True
        )

    @agent
    def coder_cloud(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_cloud"], llm=self.llm, verbose=True
        )

    @agent
    def coder_security(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_security"], llm=self.llm, verbose=True
        )

    @agent
    def coder_ai(self) -> Agent:
        return Agent(config=self.agents_config["coder_ai"], llm=self.llm, verbose=True)

    @agent
    def project_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["project_manager"], llm=self.llm, verbose=True
        )

    @agent
    def coder_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_manager"], llm=self.llm, verbose=True
        )

    @agent
    def coder_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_analyst"], llm=self.llm, verbose=True
        )

    @agent
    def coder_architect(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_architect"], llm=self.llm, verbose=True
        )

    @agent
    def coder_team_lead(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_team_lead"], llm=self.llm, verbose=True
        )

    @agent
    def coder_fullstack(self) -> Agent:
        return Agent(
            config=self.agents_config["coder_fullstack"], llm=self.llm, verbose=True
        )

    @task
    def code_writer_task(self) -> Task:
        return Task(
            config=self.tasks_config["code_writer_task"],
        )

    @task
    def code_reviewer_task(self) -> Task:
        return Task(
            config=self.tasks_config["code_reviewer_task"],
            output_file="report.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewaiPair crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            llm=self.llm,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
