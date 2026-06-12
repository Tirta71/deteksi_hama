from flask import current_app, render_template, send_from_directory


def show_index():
    return render_template("index.html")


def send_uploaded_file(filename):
    return send_from_directory(
        current_app.config["UPLOAD_FOLDER"],
        filename,
    )
