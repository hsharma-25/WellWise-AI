from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from split_docs import chunks

load_dotenv()

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

##load the existing vector store
vector_store = Chroma(
    collection_name="offset-well-reports",
    persist_directory="data/chroma_db",
    embedding_function=embedding_model,
)

##configure the retriever, use the vector store for this
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 15,
        "lambda_mult": 0.5
    }
)

##pydantic output formatting
class Finding(BaseModel):
    finding: str = Field(
        description="A factual finding or corrective action supported by evidence."
    )
    well_numbers: list[str] = Field(
        description=(
            "Well or wellbore identifiers explicitly supported by the evidence. "
            "Use an empty list if the exact well cannot be established."
        )
    )
    source_ids: list[str] = Field(
        description="Supporting source IDs, such as S1 or S2."
    )


class RAGAnswer(BaseModel):
    summary: str = Field(
        description="A concise answer to the user's question."
    )
    findings: list[Finding] = Field(
        description="Individual evidence-supported findings."
    )
    insufficient_evidence: bool = Field(
        description="True if the retrieved evidence cannot fully answer the question."
    )

##initiate the LLM
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

llm = chat_model.with_structured_output(RAGAnswer)

##prompt
prompt = PromptTemplate(
    template="""
    You are an evidence-grounded assistant for drilling operations.

    Answer the user's question using ONLY the supplied context.

    Rules:
    - Do not invent facts, causes, downtime values, or recommendations.
    - If the evidence is insufficient, explicitly say so.
    - Distinguish reported facts from interpretations.
    - Cite factual claims using the exact source labels provided,
    such as [Source 1].
    - Do not cite a source unless it supports the claim.
    - If the sources do not establish a definitive cause, say so.
    - If sources disagree, describe the disagreement.

    Context:
    {context}

    Question:
    {question}

    Provide a clear, concise answer with source citations.""",
    input_variables=['context', 'question']
)

# def format_docs(retrieved_docs):
#     context_text = "\n\n".join(docs.page_content for docs in retrieved_docs)
#     return context_text

##convert retrieved documents from vector store into context input for the prompt
# def format_docs(retrieved_docs):
#     formatted_docs = []

#     for i, doc in enumerate(retrieved_docs, start=1):
#         source = doc.metadata.get(
#             "source",
#             doc.metadata.get("file_path", "Unknown source"),
#         )

#         page = doc.metadata.get("page")
#         pdf_page = page + 1 if isinstance(page, int) else "Unknown"

#         formatted_docs.append(
#             f"[Source {i}: {source}, PDF page {pdf_page}]\n"
#             f"{doc.page_content}"
#         )

#     return "\n\n".join(formatted_docs)

def format_docs(retrieved_docs):
    formatted_docs = []

    for i, doc in enumerate(retrieved_docs, start=1):
        source_id = f"S{i}"

        source = doc.metadata.get(
            "source",
            doc.metadata.get("file_path", "Unknown source"),
        )

        page = doc.metadata.get("page")
        pdf_page = page + 1 if isinstance(page, int) else "Unknown"

        # Include report title if available; it may contain well identifiers.
        title = doc.metadata.get("title", "Unknown report title")

        formatted_docs.append(
            f"[{source_id}]\n"
            f"Source file: {source}\n"
            f"Report title: {title}\n"
            f"PDF page: {pdf_page}\n"
            f"Passage:\n{doc.page_content}"
        )

    return "\n\n".join(formatted_docs)



##output parser
parser = StrOutputParser()


# generation_chain = prompt | llm | parser


# # 6. Build the RAG chain using LCEL
# retrieval_chain = RunnableParallel(
#     {
#         "docs": retriever,
#         "question": RunnablePassthrough(),
#     }
# )

# rag_chain = (
#     retrieval_chain
#     | RunnablePassthrough.assign(
#         context=RunnableLambda(lambda x: format_docs(x["docs"]))
#     )
#     | RunnablePassthrough.assign(
#         answer=generation_chain
#     )
# )

##chain
prompt_input_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

final_chain = prompt_input_chain | prompt | llm 


if __name__ == "__main__":
    question = (
        "What are the different well no. available?"
    )

    result = final_chain.invoke(question)

    print("\n--- SUMMARY ---\n")
    print(result.summary)

    print("\n--- FINDINGS ---\n")
    for finding in result.findings:
        print(f"Finding: {finding.finding}")
        print(f"Well number(s): {finding.well_numbers or ['Not established']}")
        print(f"Sources: {finding.source_ids}")
        print()

    print("--- INSUFFICIENT EVIDENCE ---")
    print(result.insufficient_evidence)
