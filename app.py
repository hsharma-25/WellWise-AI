import streamlit as st

from src.rag_chain import final_chain


# Page configuration
st.set_page_config(
    page_title="Offset Well AI",
    page_icon="🛢️",
    layout="wide",
)

# Application heading
st.title("🛢️ Offset Well AI")
st.caption(
    "Evidence-grounded knowledge and decision support "
    "for drilling operations"
)

st.divider()

# Question input
st.subheader("Ask the Knowledge Base")

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: What drilling problems were reported "
        "in well 15/9-F-11?"
    ),
    height=100,
)

# Process the question
if st.button("Ask Question", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")

    else:
        with st.spinner("Searching reports and generating an answer..."):
            try:
                answer = final_chain.invoke(question)

                st.subheader("Answer")
                st.write(answer.summary)

                st.subheader("Findings")

                if answer.findings:
                    for i, finding in enumerate(
                        answer.findings, start=1
                    ):
                        with st.container(border=True):
                            st.markdown(f"**Finding {i}**")
                            st.write(finding.finding)

                            if finding.well_numbers:
                                st.write(
                                    "**Wells:** "
                                    + ", ".join(finding.well_numbers)
                                )

                            if finding.downtime_hours is not None:
                                st.write(
                                    "**Downtime:** "
                                    f"{finding.downtime_hours} hours"
                                )

                            if finding.source_ids:
                                st.write(
                                    "**Sources:** "
                                    + ", ".join(finding.source_ids)
                                )
                else:
                    st.info("No findings were returned.")

                if answer.insufficient_evidence:
                    st.warning(
                        "The retrieved evidence may not fully "
                        "answer the question."
                    )

            except Exception as e:
                st.error(f"Unable to generate an answer: {e}")
