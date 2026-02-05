def qa_agent(llm, context, question):

    prompt = f"""
Answer strictly from the document.

Context:
{context}

Question:
{question}

If not found, say:
"The document does not contain this information."
"""

    return llm.invoke(prompt).content
