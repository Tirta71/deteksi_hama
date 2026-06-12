import os

from flask import Flask
from ultralytics import YOLO

from simasha.config import Config
from simasha.database import init_db
from simasha.routes.auth_routes import register_auth_routes
from simasha.routes.detection_routes import register_detection_routes
from simasha.routes.web_routes import register_web_routes
from simasha.utils.file_utils import ensure_runtime_dirs


def create_app(config_class=Config):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "templates"),
        static_folder=os.path.join(base_dir, "static"),
    )
    app.config.from_object(config_class)

    ensure_runtime_dirs(app.config)
    init_db(app)
    app.extensions["yolo_model"] = YOLO(app.config["YOLO_WEIGHTS"])

    register_auth_routes(app)
    register_web_routes(app)
    register_detection_routes(app)

    return app
