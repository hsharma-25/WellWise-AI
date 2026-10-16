from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

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
        "k": 6,
        "fetch_k": 30,
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
    downtime_hours: Optional[float] = Field(
        description=(
            "Explicitly reported downtime in hours."
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
    - Cite factual claims using the exact source labels provided, such as [Source 1].
    - Do not cite a source unless it supports the claim.
    - If the sources do not establish a definitive cause, say so.
    - If sources disagree, describe the disagreement.
    - For questions asking for incidents ranked by downtime:

        - Examine every supplied passage for individual incidents with an explicitly reported downtime in hours.

        - Extract each incident's description and downtime value before selecting the results.

        - Interpret comma decimals such as 3,5 as 3.5 hours.

        - Compare the extracted values numerically and return the requested number of incidents in descending order of downtime.

        - Do not skip an incident with a higher downtime in favor of one with a lower downtime.

        - Include the downtime value in every finding.

        - Assign a well number only when the supplied evidence supports that specific incident-to-well association. Report-level well metadata alone is insufficient when a report covers multiple wells.

        - If the supplied passages do not contain enough eligible incidents, return fewer results and explain the evidence limitation. Never invent missing values.

    Context:
    {context}

    Question:
    {question}

    Provide a clear, concise answer with source citations.""",
    input_variables=['context', 'question']
)

def format_docs(retrieved_docs):
    formatted_docs = []

    for i, doc in enumerate(retrieved_docs, start=1):
        source_id = f"S{i}"

        source = doc.metadata.get(
            "report_name",
            doc.metadata.get("source", "Unknown source"),
        )

        page = doc.metadata.get("page")
        pdf_page = page + 1 if isinstance(page, int) else "Unknown"

        title = doc.metadata.get("title", "Unknown report title")
        well_numbers = doc.metadata.get("well_numbers", "Unknown")

        formatted_docs.append(
            f"[{source_id}]\n"
            f"Source file: {source}\n"
            f"Report title: {title}\n"
            f"Report-associated wells: {well_numbers}\n"
            f"PDF page: {pdf_page}\n"
            f"Passage:\n{doc.page_content}"
        )

    return "\n\n".join(formatted_docs)

##chain
prompt_input_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

final_chain = prompt_input_chain | prompt | llm 

# if __name__ == "__main__":
#     question = "Based on the retrieved passages, identify the three incidents with the highest explicitly reported downtime. Include the well identifier, downtime in hours, and source ID. Do not rank incidents when their downtime is not available."

#     retrieved_docs = retriever.invoke(question)

#     print("\nUnique reports:", len({
#         doc.metadata.get("report_name")
#         for doc in retrieved_docs
#     }))

#     print("Unique well metadata values:", len({
#         doc.metadata.get("well_numbers")
#         for doc in retrieved_docs
#     }))

#     for i, doc in enumerate(retrieved_docs, start=1):
#         print(f"\n--- S{i} ---")
#         print("Report:", doc.metadata.get("report_name"))
#         print("Title:", doc.metadata.get("title"))
#         print("Well numbers:", doc.metadata.get("well_numbers"))
#         print("PDF page:", doc.metadata.get("page", -1) + 1)
#         print("Passage:", doc.page_content)

#     answer = final_chain.invoke(question)
#     print("\n--- RAG ANSWER ---")
#     print(answer.model_dump_json(indent=2))

