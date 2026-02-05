def ats_agent(llm, resume_text, jd_text):

    prompt = f"""
You are an ATS system.

Resume:
{resume_text}

Job Description:
{jd_text}

Return:
- ATS score %
- Matching skills
- Missing skills
- Suggestions
"""

    return llm.invoke(prompt).content
