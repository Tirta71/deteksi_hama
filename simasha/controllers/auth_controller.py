import re

from flask import flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from simasha.services.user_service import create_user, get_user_by_email
from simasha.utils.auth_utils import is_logged_in


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8


def _safe_next_url(next_url):
    if next_url and next_url.startswith("/") and not next_url.startswith("//"):
        return next_url

    return url_for("index")


def show_login():
    if is_logged_in():
        return redirect(url_for("index"))

    return render_template("login.html")


def handle_login():
    if is_logged_in():
        return redirect(url_for("index"))

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    next_url = _safe_next_url(request.args.get("next"))
    errors = {}

    if not email:
        errors["email"] = "Email wajib diisi."
    elif not EMAIL_RE.match(email):
        errors["email"] = "Format email tidak valid."

    if not password:
        errors["password"] = "Password wajib diisi."

    if errors:
        return render_template(
            "login.html",
            errors=errors,
            email=email,
        ), 400

    user = get_user_by_email(email)

    if user is None or not check_password_hash(user["password_hash"], password):
        errors["form"] = "Email atau password salah."
        return render_template(
            "login.html",
            errors=errors,
            email=email,
        ), 401

    session.clear()
    session["user_id"] = user["id"]
    session["user_name"] = user["name"]
    session["user_email"] = user["email"]
    session["user_role"] = user["role"]

    flash("Login berhasil.", "success")
    return redirect(next_url)


def show_register():
    if is_logged_in():
        return redirect(url_for("index"))

    return render_template("register.html")


def handle_register():
    if is_logged_in():
        return redirect(url_for("index"))

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")
    errors = {}

    if not name:
        errors["name"] = "Nama wajib diisi."
    elif len(name) > 100:
        errors["name"] = "Nama maksimal 100 karakter."

    if not email:
        errors["email"] = "Email wajib diisi."
    elif len(email) > 150:
        errors["email"] = "Email maksimal 150 karakter."
    elif not EMAIL_RE.match(email):
        errors["email"] = "Format email tidak valid."
    elif get_user_by_email(email) is not None:
        errors["email"] = "Email sudah digunakan."

    if not password:
        errors["password"] = "Password wajib diisi."
    elif len(password) < MIN_PASSWORD_LENGTH:
        errors["password"] = "Password minimal 8 karakter."

    if not confirm_password:
        errors["confirm_password"] = "Konfirmasi password wajib diisi."
    elif password != confirm_password:
        errors["confirm_password"] = "Konfirmasi password tidak sama."

    if errors:
        return render_template(
            "register.html",
            errors=errors,
            name=name,
            email=email,
        ), 400

    password_hash = generate_password_hash(password)
    _, create_error = create_user(name, email, password_hash)

    if create_error:
        errors["email"] = create_error
        return render_template(
            "register.html",
            errors=errors,
            name=name,
            email=email,
        ), 409

    flash("Registrasi berhasil. Silakan login.", "success")
    return redirect(url_for("login"))


def handle_logout():
    session.clear()
    flash("Kamu sudah logout.", "info")
    return redirect(url_for("login"))
