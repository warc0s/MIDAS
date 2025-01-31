from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class PlotV1():
    """PlotV1 crew"""
    
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def code_writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['code_writer_agent'],
            verbose=True
        )

    @task
    def code_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config['code_generation_task'],
            agent=self.code_writer_agent(),
            output_file='grafica.py'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the PlotV1 crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
			planning=True,
			planning_llm="gemini/gemini-2.0-flash-exp"
        )