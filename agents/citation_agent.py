def citation_agent(llm, context, question):

    prompt = f"""
Answer with references.

Context:
{context}

Question:
{question}

Format:

Answer...

References:
- Section 1
- Section 2
"""

    return llm.invoke(prompt).content
