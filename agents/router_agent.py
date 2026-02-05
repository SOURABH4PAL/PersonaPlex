def route_question(question: str):

    q = question.lower()

    if "summary" in q or "summarize" in q:
        return "summary"

    if "ats" in q or "resume" in q or "score" in q:
        return "ats"

    if "explain" in q or "meaning" in q or "theme" in q:
        return "explanation"

    if "reference" in q or "citation" in q:
        return "citation"

    return "qa"
