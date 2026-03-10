import psycopg2
import uuid


# ═══════════════════════════════════════════
# CONNECTION
# ═══════════════════════════════════════════

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="qp_engine",
        user="postgres",
        password="qp2026",
        port=5432
    )


# ═══════════════════════════════════════════
# INIT DB
# ═══════════════════════════════════════════

def init_db():

    conn = get_connection()
    cur = conn.cursor()

    # ---- USERS ----
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('faculty','admin')),
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    # ---- SUBJECTS ----
    cur.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            created_by TEXT REFERENCES users(id) ON DELETE SET NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    # ---- MATERIALS ----
    cur.execute("""
        CREATE TABLE IF NOT EXISTS materials (
            id TEXT PRIMARY KEY,
            subject_id TEXT REFERENCES subjects(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            uploaded_by TEXT,
            uploaded_at TIMESTAMP DEFAULT NOW()
        )
    """)

    # ---- PAPERS ----
    cur.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            subject_id TEXT REFERENCES subjects(id) ON DELETE SET NULL,
            created_by TEXT,
            title TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    # ---- QUESTIONS ----
    cur.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id TEXT PRIMARY KEY,
            paper_id TEXT REFERENCES papers(id) ON DELETE CASCADE,
            section_name TEXT,
            question_text TEXT NOT NULL,
            bloom_level TEXT,
            difficulty TEXT,
            marks INTEGER,
            question_type TEXT,
            similarity FLOAT DEFAULT 0.0,
            source TEXT,
            is_locked BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    conn.commit()

    # ---- SEED USERS ----
    from auth import hash_password

    users = [
        (
            "user-faculty-001",
            "faculty@demo.local",
            "Faculty User",
            "faculty",
            hash_password("password123"),
        ),
        (
            "user-admin-001",
            "admin@demo.local",
            "Admin User",
            "admin",
            hash_password("admin123"),
        ),
    ]

    for u in users:
        cur.execute("SELECT id FROM users WHERE email = %s", (u[1],))
        if cur.fetchone() is None:
            cur.execute("""
                INSERT INTO users (id,email,name,role,password)
                VALUES (%s,%s,%s,%s,%s)
            """, u)
            print(f"[SEED] Created user: {u[1]}")

    conn.commit()
    cur.close()
    conn.close()

    print("[DB] Tables initialized")


# ═══════════════════════════════════════════
# USER OPERATIONS
# ═══════════════════════════════════════════

def get_user_by_email(email: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id,email,name,role,password FROM users WHERE email=%s",
        (email,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return {"id": row[0], "email": row[1], "name": row[2], "role": row[3], "password": row[4]}
    return None


# ═══════════════════════════════════════════
# SUBJECT OPERATIONS
# ═══════════════════════════════════════════

def create_subject(name: str, description: str, created_by: str) -> str:
    conn = get_connection()
    cur = conn.cursor()
    subject_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO subjects (id, name, description, created_by)
        VALUES (%s, %s, %s, %s)
    """, (subject_id, name, description, created_by))
    conn.commit()
    cur.close()
    conn.close()
    return subject_id


def get_all_subjects():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.id, s.name, s.description, s.created_at, u.name as created_by_name,
               COUNT(DISTINCT m.id) as material_count,
               COUNT(DISTINCT p.id) as paper_count
        FROM subjects s
        LEFT JOIN users u ON s.created_by = u.id
        LEFT JOIN materials m ON m.subject_id = s.id
        LEFT JOIN papers p ON p.subject_id = s.id
        GROUP BY s.id, s.name, s.description, s.created_at, u.name
        ORDER BY s.created_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "id": r[0], "name": r[1], "description": r[2],
            "created_at": str(r[3]), "created_by": r[4],
            "material_count": r[5], "paper_count": r[6]
        }
        for r in rows
    ]


def get_subject_by_id(subject_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, description FROM subjects WHERE id=%s", (subject_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return {"id": row[0], "name": row[1], "description": row[2]}
    return None


def delete_subject(subject_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM subjects WHERE id=%s", (subject_id,))
    conn.commit()
    cur.close()
    conn.close()


# ═══════════════════════════════════════════
# MATERIAL OPERATIONS
# ═══════════════════════════════════════════

def insert_material(subject_id: str, filename: str, file_path: str, uploaded_by: str) -> str:
    conn = get_connection()
    cur = conn.cursor()
    material_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO materials (id, subject_id, filename, file_path)
        VALUES (%s, %s, %s, %s)
    """, (material_id, subject_id, filename, file_path))
    conn.commit()
    cur.close()
    conn.close()
    return material_id


def get_materials_by_subject(subject_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, subject_id, filename, file_path, uploaded_at
        FROM materials WHERE subject_id=%s ORDER BY uploaded_at DESC
    """, (subject_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "id": r[0], "subject_id": r[1], "filename": r[2],
            "file_path": r[3], "uploaded_at": str(r[4])
        }
        for r in rows
    ]


def get_all_materials():
    """Used on startup to reload all materials into memory."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, subject_id, filename, file_path FROM materials ORDER BY uploaded_at ASC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"id": r[0], "subject_id": r[1], "filename": r[2], "file_path": r[3]} for r in rows]


def delete_material(material_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM materials WHERE id=%s", (material_id,))
    conn.commit()
    cur.close()
    conn.close()


# ═══════════════════════════════════════════
# PAPER OPERATIONS
# ═══════════════════════════════════════════

def create_paper(subject_id: str = None, created_by: str = None, title: str = None) -> str:
    conn = get_connection()
    cur = conn.cursor()
    paper_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO papers (id, subject_id, created_by, title)
        VALUES (%s, %s, %s, %s)
    """, (paper_id, subject_id, created_by, title))
    conn.commit()
    cur.close()
    conn.close()
    return paper_id


def get_papers_by_subject(subject_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT p.id, p.title, p.created_at, p.created_by,
               COUNT(q.id) as question_count
        FROM papers p
        LEFT JOIN questions q ON q.paper_id = p.id
        WHERE p.subject_id = %s
        GROUP BY p.id, p.title, p.created_at, p.created_by
        ORDER BY p.created_at DESC
    """, (subject_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "id": r[0], "title": r[1], "created_at": str(r[2]),
            "created_by": r[3], "question_count": r[4]
        }
        for r in rows
    ]


def get_paper_with_questions(paper_id: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT id, title, subject_id, created_at FROM papers WHERE id=%s", (paper_id,))
    paper_row = cur.fetchone()
    if not paper_row:
        cur.close()
        conn.close()
        return None

    cur.execute("""
        SELECT id, section_name, question_text, bloom_level, difficulty,
               marks, question_type, similarity, source
        FROM questions WHERE paper_id=%s ORDER BY section_name, created_at
    """, (paper_id,))
    q_rows = cur.fetchall()

    cur.close()
    conn.close()

    sections = {}
    for q in q_rows:
        sec = q[1] or "A"
        if sec not in sections:
            sections[sec] = []
        sections[sec].append({
            "id": q[0], "text": q[2], "bloom": q[3], "difficulty": q[4],
            "marks": q[5], "question_type": q[6], "similarity": q[7], "source": q[8]
        })

    return {
        "id": paper_row[0],
        "title": paper_row[1],
        "subject_id": paper_row[2],
        "created_at": str(paper_row[3]),
        "sections": [{"name": k, "questions": v} for k, v in sections.items()]
    }


# ═══════════════════════════════════════════
# QUESTION OPERATIONS
# ═══════════════════════════════════════════

def insert_question(
    paper_id, section_name, question_text,
    bloom_level, difficulty, marks,
    similarity, source, question_type="Short Answer"
):
    conn = get_connection()
    cur = conn.cursor()
    question_id = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO questions
        (id, paper_id, section_name, question_text, bloom_level, difficulty, marks, similarity, source, question_type)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        question_id, paper_id, section_name, question_text,
        bloom_level, difficulty, marks, similarity, source, question_type
    ))
    conn.commit()
    cur.close()
    conn.close()
    return question_id


def update_question_text(question_id, new_text):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE questions SET question_text=%s WHERE id=%s", (new_text, question_id))
    conn.commit()
    cur.close()
    conn.close()


def delete_question_by_id(question_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM questions WHERE id=%s", (question_id,))
    conn.commit()
    cur.close()
    conn.close()


# ═══════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════

def get_analytics_summary():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM papers")
    papers = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM questions")
    questions = cur.fetchone()[0]
    cur.execute("SELECT AVG(similarity) FROM questions")
    avg_sim = cur.fetchone()[0] or 0
    cur.close()
    conn.close()
    return {
        "total_papers_generated": papers,
        "total_questions_generated": questions,
        "repetition_percentage": round(avg_sim * 100, 2),
        "ai_assist_percentage": 100 if questions > 0 else 0
    }


def get_bloom_distribution():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT bloom_level, COUNT(*) FROM questions GROUP BY bloom_level")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"level": r[0], "count": r[1]} for r in rows]


def get_difficulty_distribution():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT difficulty, COUNT(*) FROM questions GROUP BY difficulty")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"level": r[0], "count": r[1]} for r in rows]