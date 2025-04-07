import dotenv   
dotenv.load_dotenv()

import streamlit as st
import os
from smolagents import HfApiModel, CodeAgent, LiteLLMModel, FinalAnswerStep, PlanningStep
from agentic.recipe_retrieval_tool import RetrieverTool
from agentic.web_search_tool import visit_webpage
from pprint import pprint
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    HfApiModel,
    DuckDuckGoSearchTool,
    LiteLLMModel,
)
import litellm

st.set_page_config(
    page_title="Materials Agent",
    page_icon=":microscope:",
    layout="wide",
)

st.title("Materials Agent")

openai_api_key = st.text_input("OpenAI API Key", os.environ.get("OPENAI_API_KEY", "empty"), type="password")

if openai_api_key == "empty":
    st.warning("Please enter your OpenAI API key to use the model.")
    st.stop()


@st.cache_resource
def get_agent():
    model = LiteLLMModel(
        model_id="o3-mini",
        max_completion_tokens=16384,
        api_key=openai_api_key,
        # reasoning_effort="high"
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
    return agent


agent = get_agent()

if st.session_state.get("messages") is None:
    st.session_state.messages = []

with st.sidebar:
    clear_btn = st.button("Clear Conversation")
    if clear_btn:
        st.session_state.messages = []

def display(step):
    with st.chat_message("assistant"):
        if isinstance(step, FinalAnswerStep):
            st.markdown(step.final_answer)
        elif isinstance(step, PlanningStep):
            st.markdown(f"Planning step: {step.plan}")
        else:
            if "Code:" in step.model_output:
                thought, code = step.model_output.split("Code:\n", 1)
                st.markdown(thought)
                with st.expander("Action", expanded=False):
                    st.markdown(code)
                with st.expander("Observations", expanded=False):
                    st.markdown(step.observations)
            else:
                st.markdown(step.model_output)

            

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        display(message["content"])

prompt = st.chat_input("Ask a question about materials synthesis or provide a material name and synthesis technique.")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    instruction = f"""Follow the user's request to predict the recipe to synthesize the following material:
# User Request
{prompt}

utilize the recipe_retrieval and web_search tools to find the most relevant recipes.

# Response Guide
1. Use more than three search queries to get as much ground as possible using the recipe_retrieval and web_search tools.
2. Consider the another methods to synthesize the material.
3. Collect as much information as possible about the material and its synthesis.
4. Provide the final answer in a clear and concise format including at least three retrieved recipes.

## Citation Guide
Your final answer in a structured format with proper citations in the following format:
1. All factual information must be cited inline format [number].
2. List the citations at the end of the answer in a separate section titled "## References".
3. The citation format should be [number] Document Title (Author, Year), Website Name, URL.
4. For example: "According to Wikipedia[...][1]"
"""
    with st.spinner("Generating response..."):
        for step in agent.run(instruction, stream=True, reset=False):
            display(step)
            st.session_state.messages.append({
                "role": "assistant",
                "content": step
            })