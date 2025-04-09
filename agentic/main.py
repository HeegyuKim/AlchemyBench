import dotenv   
import os
from smolagents import HfApiModel, CodeAgent, LiteLLMModel, FinalAnswerStep, PlanningStep
from recipe_retrieval_tool import RetrieverTool
from web_search_tool import visit_webpage
from pprint import pprint
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    HfApiModel,
    DuckDuckGoSearchTool,
    LiteLLMModel,
)


dotenv.load_dotenv()

model = LiteLLMModel(
    model_id="o3-mini",
    max_completion_tokens=16384,
)

database_search_agent = ToolCallingAgent(
    tools=[RetrieverTool()],
    model=model,
    max_steps=10,
    name="database_search_agent",
    description="Retrieves recipes for materials synthesis from a database. Provide research questions to find relevant knowledge, not just keywords.",
)
database_search_agent.prompt_templates["managed_agent"]["task"] += """
Select the relevant recipes from the retrieved results and provide helpful knowledge in `final_answer()` to answer the user's query.
If the retrieved recipes are not relevant, try another query to find more relevant recipes.

Additionally, provide `### 4. References` at the end of your output.
This section should include the references to the original sources of the knowledge you provided including explanations why they are relevant.
"""

web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), visit_webpage],
    model=model,
    max_steps=10,
    name="web_search_agent",
    description="Retrieves recipes for materials synthesis from the web search engine. Provide research questions to find relevant knowledge, not just keywords.",
)
web_agent.prompt_templates["managed_agent"]["task"] += """
Select the relevant recipes from the retrieved results and provide helpful knowledge in `final_answer()` to answer the user's query.
If the retrieved recipes are not relevant, try another query to find more relevant recipes.

Additionally, provide `### 4. References` at the end of your output.
This section should include the references to the original sources of the knowledge you provided including explanations why they are relevant.
"""

agent = CodeAgent(
    tools=[],
    model=model,
    max_steps=20,
    planning_interval=4,
    # verbosity_level=0, 
    managed_agents=[database_search_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)


instruction = "Predict the recipe to synthesize the following material:"

target = f"""{instruction}

## Key Contributions
- Novel materials or compounds: Layered-oxide LiCoO2 (LCO) cathode materials for lithium-ion batteries.
- Unique synthesis methods: Synthesis via aerosol spray pyrolysis using nitrate and acetate metal precursors.
- Specific applications or domains: High-performance cathode materials for lithium-ion batteries."""

target += """

# Guide to the Answer
1. Utilize the web_search_agent and database_search_agent tool to find knowledges, try diverse research questions to cover different aspects of the target material.
2. Analyze the retrieved knowledge to identify common synthesis methods and materials used. 
3. If the retrieved knowledges are insufficient, try another research questions.
4. Provide a detailed synthesis procedure, including specific conditions and parameters.
5. Include evidence and references to support the proposed recipe.

# Guide to submit final answer
- First, use `database_search_agent` to find knowledge from the database.
- Then I will give you the knowledge retrieved from the database.
- If the knowledge is not relevant, try another query to find more relevant recipes.
- Do not submit your answer until you are confident it is correct. Keep trying until you are confident.
"""

# Run the agent
# result = agent.run(target)
# print("Agent's response:")
# print(result)

print(agent.run(target))

# for step in agent.run(target, stream=True):
#     if isinstance(step, FinalAnswerStep):
#         print("Final answer:")
#         print(step.final_answer)
#         break
#     if isinstance(step, PlanningStep):
#         print(f"Planning step: {step.plan}")
#     else:
#         print(f"Action step: {step.step_number}")
#         print(step.model_output)
#         print("\n" + "=" * 50 + "\n")
