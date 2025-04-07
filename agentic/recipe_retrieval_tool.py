from smolagents import Tool
from datasets import load_dataset, load_from_disk, concatenate_datasets
import os
import numpy as np
from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever


class RetrieverTool(Tool):
    name = "recipe_retriever"
    description = "Uses semantic search to retrieve the parts of the Open Materials Guide (OMG) dataset that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target recipes. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, retrieval_split = "train", rag_topk: int = 10, **kwargs):
        super().__init__(**kwargs)
        
        if retrieval_split == "all":
            retrieval_set = load_dataset("iknow-lab/open-materials-guide-0210-embeddings")
            knowledge_base = concatenate_datasets(retrieval_set.values())
        else:
            knowledge_base = load_dataset("iknow-lab/open-materials-guide-0210-embeddings", split="train")
        self.rag_topk = rag_topk
        
        docs = [
            Document(page_content="{contribution}\n\n{recipe}".format(**doc), metadata={"id": doc["id"]})
            for doc in knowledge_base
        ]

        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=500,
        #     chunk_overlap=50,
        #     add_start_index=True,
        #     strip_whitespace=True,
        #     separators=["\n\n", "\n", ".", " ", ""],
        # )
        # docs_processed = text_splitter.split_documents(source_docs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=rag_topk
        )



    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {doc.metadata['id'][:10]} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )



if __name__ == "__main__":
    retriever_tool = RetrieverTool()

    query = "How to make a ZnO thin film?"
    result = retriever_tool(query)
    print(result)