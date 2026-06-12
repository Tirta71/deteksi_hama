from functools import wraps

from flask import jsonify, redirect, request, session, url_for


def is_logged_in():
    return bool(session.get("user_id"))


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if is_logged_in():
            return view(*args, **kwargs)

        if request.path.startswith("/detect"):
            return jsonify({
                "status": "error",
                "message": "Silakan login terlebih dahulu.",
            }), 401

        return redirect(url_for("login", next=request.path))

    return wrapped_view
