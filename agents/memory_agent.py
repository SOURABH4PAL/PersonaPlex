def get_context(vectorstore, question, k=5):

    if vectorstore is None:
        return ""

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    return "\n\n".join([d.page_content for d in docs])
