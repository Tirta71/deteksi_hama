from flask import render_template, session

from simasha.services.history_service import get_detection_rows


def show_detection_history():
    rows = get_detection_rows(
        user_id=session.get("user_id"),
        include_all=session.get("user_role") == "admin",
    )

    return render_template(
        "hasil_deteksi.html",
        rows=rows,
    )
