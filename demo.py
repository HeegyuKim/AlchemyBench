from dotenv import load_dotenv

load_dotenv()


import streamlit as st
import os
from experiment.predict import RAGRecipePredictor, RecipePredictor
from litellm import embedding
import litellm
from pdf2recipe import pdf_bytelist_to_recipes
from litellm import completion

st.set_page_config(
    page_title="Materials Synthesis Recipe Recommender",
    page_icon=":microscope:",
    layout="wide",
)

st.title("Materials Synthesis Recipe Recommender")
# Input fields
st.sidebar.title("Input Parameters")

openai_key = st.session_state.get("openai_key", os.environ.get("OPENAI_API_KEY", "empty"))
openai_api_key = st.sidebar.text_input("OpenAI API Key", openai_key, type="password")
update_key = st.sidebar.button("Update Key")
if update_key:
    st.session_state.openai_key = openai_api_key
    litellm.openai_key = openai_api_key
    litellm.api_key = openai_api_key
    st.toast("API Key updated successfully")
    st.rerun()

with st.sidebar, st.form("recipe_form"):
    material_name = st.text_input("Material Name", "ZnO")
    synthesis_technique = st.text_input("Synthesis Technique", "Solution-based")
    application = st.text_input("Application", "Photocatalysis")
    other_contstraints = st.text_area("Other Constraints", "")

    top_k = st.slider("Number of Retrievals", 0, 10, 5)
    model = st.selectbox("Model", ["gpt-4o-mini", "o3-mini-high", "o1", "gpt-4o-2024-11-20"])

    files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
    if files:
        for file in files:
            st.write(file.name)

    generate_btn = st.form_submit_button("Recommend")

clear_btn = st.sidebar.button("Clear Conversation")
if clear_btn:
    del st.session_state.messages
    del st.session_state.references
    st.rerun()
    
if not generate_btn and not hasattr(st.session_state, "messages"):
    st.write("This is a demo of the Materials Synthesis Recipe Recommender. Please enter the desired material properties and click on the 'Recommend' button to get a list of materials synthesis recipes that can be used to synthesize materials with the desired properties.")
    st.stop()

use_rag = top_k >= 1
output_filename = f"data/recipes.jsonl"

PREDICTION_PROMPT = """## Key Contributions
- **Novel materials or compounds**: {material_name}
- **Unique synthesis methods**: {synthesis_technique}
- **Specific applications or domains**: {application}
""".strip()

def get_embedding(contributions):
    response = embedding(model='text-embedding-3-large', input=[contributions])
    emb = response['data'][0]['embedding']
    return emb


@st.cache_resource
def get_predictors():
    rag_predictor = RAGRecipePredictor(model=model, prompt_filename="experiment/prompts/rag.txt", rag_topk=top_k, retrieval_split="all", api_key=openai_api_key)
    base_predictor = RecipePredictor(model=model, prompt_filename="experiment/prompts/prediction.txt", api_key=openai_api_key)
    return rag_predictor, base_predictor

rag_predictor, base_predictor = get_predictors()

def predict_recipe(material_name, synthesis_technique, application, other_contstraints, top_k, model, use_rag, files=None):
    contributions = PREDICTION_PROMPT.format(
        material_name=material_name,
        synthesis_technique=synthesis_technique,
        application=application,
    )

    if use_rag or files:
        predictor = rag_predictor
        emb = get_embedding(contributions)
    else:
        predictor = base_predictor
        emb = None

    if files:
        with st.spinner("Extracting recipes from PDFs..."):
            references = pdf_bytelist_to_recipes([file.read() for file in files])
    else:
        references = None
    
    predictor.base_references = references
    predictor.model = model

    if other_contstraints:
        contributions += f"\n\n## Other Constraints\n{other_contstraints}"
    batch = [
        {
            "contribution": contributions,
            "recipe": "",
            "contributions_embedding": emb
        }
    ]

    for _, output in predictor.predict(batch):
        pass

    if use_rag or files:
        ref_outputs = []
        if references:
            ref_outputs.extend(references)

        references = predictor.search(emb, k=top_k, return_rows=True)

        for i in range(top_k):
            rid, contribution, recipe = references['id'][i], references['contribution'][i], references['recipe'][i]

            ref_output = f"Semantic Scholar: [{rid}](https://www.semanticscholar.org/paper/{rid})\n"
            ref_output +=f"{contribution}\n\n{recipe}"
            ref_outputs.append(ref_output)

        references = ref_outputs
    else:
        references = None
    
    prompt = predictor.build_prompt(batch[0])[0]['content']

    return output, references, prompt
    
if not hasattr(st.session_state, "messages"):
    st.session_state.messages = []

    with st.spinner("Generating recipes..."):
        recipe, references, user_prompt = predict_recipe(material_name, synthesis_technique, application, other_contstraints, top_k, model, use_rag, files=files)
        st.session_state.references = references
        st.session_state.messages.append({
            "role": "user",
            "content": user_prompt,
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": recipe
        })
else:
    recipe, references, user_prompt = st.session_state.messages[1]["content"], st.session_state.references, st.session_state.messages[0]["content"]


with st.chat_message("assistant"):
    st.header("Predicted Recipes")
    st.markdown(recipe)
    st.write("\n\n")

    if use_rag:
        st.header("References")
        for i, ref in enumerate(references):
            with st.expander(f"Reference {i + 1}", expanded=False):
                st.markdown(ref)

if len(st.session_state.messages) > 2:
    for message in st.session_state.messages[2:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = st.chat_input("Ask a question about the recipe")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.spinner("Generating response..."):
        if model.startswith("o1") or model.startswith("o3"):
            response = completion(
                model=model,
                messages=st.session_state.messages,
                max_completion_tokens=16384,
            )
        else:
            response = completion(
                model=model,
                messages=st.session_state.messages,
                max_tokens=4096,
            )

    with st.chat_message("assistant"):
        st.markdown(response["choices"][0]["message"]["content"])
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["choices"][0]["message"]["content"]
    })