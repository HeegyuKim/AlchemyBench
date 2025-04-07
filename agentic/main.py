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
web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), visit_webpage],
    model=model,
    max_steps=10,
    name="web_search_agent",
    description="Runs web searches for you.",
)

agent = CodeAgent(
    tools=[RetrieverTool()], 
    model=model,
    max_steps=10,
    planning_interval=4,
    # verbosity_level=0, 
    managed_agents=[web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)


instruction = "Predict the recipe to synthesize the following material:"

target = f"""{instruction}

## Key Contributions
- Novel materials or compounds: Layered-oxide LiCoO2 (LCO) cathode materials for lithium-ion batteries.
- Unique synthesis methods: Synthesis via aerosol spray pyrolysis using nitrate and acetate metal precursors.
- Specific applications or domains: High-performance cathode materials for lithium-ion batteries."""

target += """

# Guide to the Prediction
1. Utilize the web_search_agent and recipe_retriever tool to find relevant recipes, try diverse queries to cover different aspects of the target material.
2. Analyze the retrieved recipes to identify common synthesis methods and materials used. 
3. If the retrieved recipes are not relevant, try another query to find more relevant recipes.
4. Predict a recipe for synthesizing the target material based on the analysis.
5. Ensure the predicted recipe is novel and not previously reported in the literature.
6. Provide a detailed synthesis procedure, including specific conditions and parameters.

# Guide to submit final answer
- Call the `final_answer()` tool to submit your final answer.
- Do not submit your answer until you are confident it is correct.
- Your final answer MUST be in the following format:


## Materials
<provide the list of materials used in the synthesis, including their chemical formulas and any specific grades or purities required.>

## Synthesis Equipment
<provide the list of equipment used in the synthesis, including any specific models or brands required.>

## Synthesis Procedure
<provide a detailed step-by-step procedure for the synthesis, including any specific conditions or parameters required.>

## Characterization Methods and Equipment
<provide the list of characterization methods and equipment used to analyze the synthesized material, including any specific models or brands required.>

## Product Characteristics
<provide a detailed description of the expected characteristics of the synthesized material, including any specific properties or performance metrics.>
"""

# Run the agent
# result = agent.run(target)
# print("Agent's response:")
# print(result)

for step in agent.run(target, stream=True):
    if isinstance(step, FinalAnswerStep):
        print("Final answer:")
        print(step.final_answer)
        break
    if isinstance(step, PlanningStep):
        print(f"Planning step: {step.plan}")
    else:
        print(f"Action step: {step.step_number}")
        print(step.model_output)
        print("\n" + "=" * 50 + "\n")
