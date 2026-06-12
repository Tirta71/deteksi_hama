def validate_upload(file_storage):
    if not file_storage or file_storage.filename == "":
        return "No file"

    return None
