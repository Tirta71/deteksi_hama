from flask import current_app, g
from mysql.connector import Error, pooling


def _create_pool():
    return pooling.MySQLConnectionPool(
        pool_name=current_app.config["DB_POOL_NAME"],
        pool_size=current_app.config["DB_POOL_SIZE"],
        host=current_app.config["DB_HOST"],
        port=current_app.config["DB_PORT"],
        database=current_app.config["DB_NAME"],
        user=current_app.config["DB_USER"],
        password=current_app.config["DB_PASSWORD"],
        charset=current_app.config["DB_CHARSET"],
        autocommit=False,
    )


def get_db():
    if "db" not in g:
        if "mysql_pool" not in current_app.extensions:
            current_app.extensions["mysql_pool"] = _create_pool()

        g.db = current_app.extensions["mysql_pool"].get_connection()

    return g.db


def close_db(error=None):
    db = g.pop("db", None)

    if db is not None and db.is_connected():
        db.close()


def init_db(app):
    app.teardown_appcontext(close_db)


def check_db_connection():
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        return True, None
    except Error as exc:
        return False, str(exc)
