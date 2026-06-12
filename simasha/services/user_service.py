from mysql.connector import IntegrityError

from simasha.database import get_db


def get_user_by_email(email):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT id, name, email, password_hash, role, created_at, updated_at
            FROM users
            WHERE email = %s
            LIMIT 1
            """,
            (email,),
        )
        return cursor.fetchone()
    finally:
        cursor.close()


def create_user(name, email, password_hash, role="user"):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            INSERT INTO users (name, email, password_hash, role)
            VALUES (%s, %s, %s, %s)
            """,
            (name, email, password_hash, role),
        )
        db.commit()
        return cursor.lastrowid, None
    except IntegrityError:
        db.rollback()
        return None, "Email sudah digunakan."
    except Exception:
        db.rollback()
        raise
    finally:
        cursor.close()
