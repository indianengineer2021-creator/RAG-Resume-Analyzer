from typing import Dict


RESUME_ANALYSIS_PROMPT = """
You are an expert technical recruiter and resume evaluator.

Analyze the candidate's resume against the provided Job Description.
Use ONLY the retrieved resume context below.
Do not invent skills, years of experience, job titles, projects, technologies, responsibilities, or certifications.
If the information is not present in the retrieved resume context, state: "Not found in resume".

Job Description:
{job_description}

Retrieved Resume Context:
{context}

Evaluate the resume across these areas:
1. Overall match
2. Technical skill match
3. Required skills found
4. Missing skills
5. Relevant experience level
6. Project relevance
7. Domain relevance
8. Certifications
9. Resume strengths
10. Resume weaknesses
11. ATS keyword gaps
12. Recommended improvements
13. Interview preparation areas

Return valid JSON in this format:
{
  "overall_score": 0,
  "summary": "...",
  "skill_match": [
    {"skill": "Python", "status": "Found", "evidence": "..."}
  ],
  "missing_skills": ["..."],
  "experience_match": "...",
  "domain_match": "...",
  "certification_match": "...",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "ats_gaps": ["..."],
  "recommendations": ["..."],
  "interview_topics": ["..."]
}

Requirements:
- overall_score must be a number between 0 and 100.
- Use "Not found in resume" when evidence is missing.
- Keep the response grounded in the retrieved resume context only.
- Do not use assumptions or general candidate profile patterns.
"""


def build_resume_analysis_prompt(job_description: str, context: str) -> str:
    """Create the final prompt sent to Gemini for grounded evaluation."""
    return RESUME_ANALYSIS_PROMPT.format(job_description=job_description, context=context)


def build_follow_up_prompt(question: str, context: str) -> str:
    """Create a grounded prompt for follow-up questions about the resume."""
    return f"""
You are answering questions about a candidate's resume using only the retrieved resume context.
Do not use assumptions or general knowledge.
If the answer is not present in the context, respond with: "Not found in resume".

User Question:
{question}

Retrieved Resume Context:
{context}

Answer clearly and briefly, and support your answer with evidence from the retrieved resume text.
"""
