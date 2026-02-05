def summary_agent(llm, context):

    prompt = f"""
Summarize the document clearly.

Document:
{context}
"""

    return llm.invoke(prompt).content
