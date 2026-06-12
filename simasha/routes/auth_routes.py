from simasha.controllers.auth_controller import (
    handle_login,
    handle_logout,
    handle_register,
    show_login,
    show_register,
)


def register_auth_routes(app):
    @app.route("/login", methods=["GET"])
    def login():
        return show_login()

    @app.route("/login", methods=["POST"])
    def login_post():
        return handle_login()

    @app.route("/register", methods=["GET"])
    def register():
        return show_register()

    @app.route("/register", methods=["POST"])
    def register_post():
        return handle_register()

    @app.route("/logout", methods=["GET"])
    def logout():
        return handle_logout()
