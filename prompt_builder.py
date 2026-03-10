"""
prompt_builder.py — Construct structured prompts for LLM generation.

Supports question types:
  - MCQ (with 4 options, correct answer marked)
  - Short Answer
  - Long Answer
  - Case-based
"""


def build_prompt(
    context_chunks,
    question_type,
    bloom_level,
    difficulty,
    marks,
    verb,
    focus
):
    context = "\n\n".join(context_chunks)

    # ── BASE CONSTRAINTS (shared across all types) ──
    base_constraints = f"""
- Bloom's level: {bloom_level}
- Difficulty: {difficulty}
- Marks: {marks}
- Question must be original and untraceable
- Do not copy sentences verbatim from the source material
- Use the question verb: "{verb}"
- Focus: {focus}
- Academic tone required
"""

    # ── MCQ-SPECIFIC PROMPT ──
    if question_type.upper() == "MCQ":
        prompt = f"""
You are an academic question paper generator.

Use ONLY the following course material:
{context}

Generate ONE Multiple Choice Question (MCQ).

Constraints:
{base_constraints}

MCQ Format Rules:
- Provide exactly 4 options labeled A), B), C), D)
- Exactly ONE option must be correct
- Wrong options (distractors) must be plausible but clearly incorrect
- Options should be of similar length and complexity
- Do not use "All of the above" or "None of the above"

Output format (follow EXACTLY):
QUESTION: <the question text>
A) <option A>
B) <option B>
C) <option C>
D) <option D>
ANSWER: <correct letter, e.g. B>
"""

    # ── CASE-BASED PROMPT ──
    elif question_type.upper() == "CASE-BASED":
        prompt = f"""
You are an academic question paper generator.

Use ONLY the following course material:
{context}

Generate ONE Case-Based question.

Constraints:
{base_constraints}

Case-Based Format Rules:
- First write a realistic case study scenario (3-5 sentences) relevant to the syllabus
- Then ask 2-3 sub-questions based on the case
- Sub-questions should test analysis and application skills
- Label sub-questions as (a), (b), (c)

Output format (follow EXACTLY):
CASE: <case study scenario>
(a) <sub-question 1>
(b) <sub-question 2>
(c) <sub-question 3>
"""

    # ── LONG ANSWER PROMPT ──
    elif question_type.upper() == "LONG ANSWER":
        prompt = f"""
You are an academic question paper generator.

Use ONLY the following course material:
{context}

Generate ONE Long Answer question.

Constraints:
{base_constraints}

Long Answer Format Rules:
- Question should require a detailed response (200-400 words expected)
- Should test deep understanding, not just recall
- May include sub-parts labeled (a), (b) if appropriate
- Should encourage critical thinking and explanation

Output only the question text.
"""

    # ── SHORT ANSWER (DEFAULT) ──
    else:
        prompt = f"""
You are an academic question paper generator.

Use ONLY the following course material:
{context}

Generate ONE {question_type} question.

Constraints:
{base_constraints}

Output only the question text.
"""

    return prompt.strip()