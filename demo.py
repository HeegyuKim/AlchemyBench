import streamlit as st
import os
from experiment.predict import RAGRecipePredictor, RecipePredictor
from litellm import embedding

st.title("Materials Synthesis Recipe Recommender")
st.write("This is a demo of the Materials Synthesis Recipe Recommender. Please enter the desired material properties and click on the 'Recommend' button to get a list of materials synthesis recipes that can be used to synthesize materials with the desired properties.")

# Input fields
st.sidebar.title("Input Parameters")

openai_api_key = st.sidebar.text_input("OpenAI API Key", os.environ.get("OPENAI_API_KEY"), type="password")
os.environ["OPENAI_API_KEY"] = openai_api_key
material_name = st.sidebar.text_input("Material Name", "ZnO")
synthesis_technique = st.sidebar.text_input("Synthesis Technique", "Solution-based")
application = st.sidebar.text_input("Application", "Photocatalysis")
other_contstraints = st.sidebar.text_area("Other Constraints", "")

top_k = st.sidebar.slider("Number of Retrievals", 1, 10, 5)
model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "o3-mini-high", "o1", "gpt-4o-2024-11-20"])

generate_btn = st.sidebar.button("Recommend")

if not generate_btn:
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
    rag_predictor = RAGRecipePredictor(model=model, prompt_filename="experiment/prompts/rag.txt", rag_topk=top_k, retrieval_split="all")
    base_predictor = RecipePredictor(model=model, prompt_filename="experiment/prompts/prediction.txt")
    return rag_predictor, base_predictor

rag_predictor, base_predictor = get_predictors()

def predict_recipe(material_name, synthesis_technique, application, other_contstraints, top_k, model, use_rag):
    contributions = PREDICTION_PROMPT.format(
        material_name=material_name,
        synthesis_technique=synthesis_technique,
        application=application,
    )

    if use_rag:
        predictor = rag_predictor
        emb = get_embedding(contributions)
    else:
        predictor = base_predictor
        emb = None

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

    if use_rag:
        references = predictor.search(emb, k=top_k, return_rows=True)
        ref_outputs = []

        for i in range(top_k):
            rid, contribution, recipe = references['id'][i], references['contribution'][i], references['recipe'][i]

            ref_output = f"Semantic Scholar: [{rid}](https://www.semanticscholar.org/paper/{rid})\n"
            ref_output +=f"{contribution}\n\n{recipe}"
            ref_outputs.append(ref_output)

        references = ref_outputs
    else:
        references = None
    
    return output, references
    
with st.spinner("Generating recipes..."):
    recipe, references = predict_recipe(material_name, synthesis_technique, application, other_contstraints, top_k, model, use_rag)

st.header("Predicted Recipes")
st.markdown(recipe)
st.write("\n\n")

if use_rag:
    st.header("References")
    for i, ref in enumerate(references):
        with st.expander(f"Reference {i + 1}", expanded=False):
            st.markdown(ref)

