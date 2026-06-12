from simasha.controllers.detection_controller import handle_detect_both
from simasha.utils.auth_utils import login_required


def register_detection_routes(app):
    @app.route("/detect_both", methods=["POST"])
    @login_required
    def detect_both():
        return handle_detect_both()
