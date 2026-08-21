def get_rag_system_prompt(context_text: str, question: str) -> str:

    """Returns the system prompt formatted with the retrieved context and user question."""

    return (
        "You are a secure corporate assistant. Use ONLY the following retrieved business context to answer "
        "the employee's question. If the answer cannot be found in the context, state clearly that you do "
        "not possess that information.\n\n"
        f"--- CONTEXT ---\n{context_text}\n----------------\n\n"
        f"QUESTION: {question}"
    )