from simasha.controllers.history_controller import show_detection_history
from simasha.controllers.web_controller import send_uploaded_file, show_index
from simasha.utils.auth_utils import login_required


def register_web_routes(app):
    @app.route("/", methods=["GET"])
    @login_required
    def index():
        return show_index()

    @app.route("/hasil_deteksi", methods=["GET"])
    @login_required
    def hasil_deteksi():
        return show_detection_history()

    @app.route("/uploads/<filename>")
    @login_required
    def uploaded_file(filename):
        return send_uploaded_file(filename)
