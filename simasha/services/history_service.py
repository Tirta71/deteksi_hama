import json
from collections import OrderedDict
from decimal import Decimal

from simasha.database import get_db


def _classes_to_text(classes):
    if not classes:
        return ""

    return ", ".join(classes)


def _classes_from_text(classes_text):
    if not classes_text or classes_text == "-":
        return []

    return [
        item.strip()
        for item in classes_text.split(",")
        if item.strip()
    ]


def _class_summary(classes):
    counts = OrderedDict()

    for class_name in classes or []:
        if not class_name:
            continue

        counts[class_name] = counts.get(class_name, 0) + 1

    if not counts:
        return "-"

    return ", ".join(
        f"{count} {class_name}"
        for class_name, count in counts.items()
    )


def _safe_int(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _format_dt(dt):
    if not dt:
        return "-"

    if isinstance(dt, str):
        return dt

    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _to_float(value):
    if value is None:
        return 0.0

    if isinstance(value, Decimal):
        return float(value)

    return value


def _serialize_per_grid_results(per_grid_results):
    safe_results = []

    for item in per_grid_results or []:
        classes = item.get("classes", []) or []

        safe_results.append({
            "grid": item.get("grid"),
            "image": item.get("image"),
            "num_boxes": _safe_int(item.get("num_boxes")),
            "classes": classes,
            "class_summary": _class_summary(classes),
            "total_pests": len(classes),
            "avg_conf": _to_float(item.get("avg_conf")),
        })

    return json.dumps(safe_results, ensure_ascii=False)


def _parse_per_grid_results(value):
    if not value:
        return []

    try:
        results = json.loads(value)
    except (TypeError, ValueError):
        return []

    if not isinstance(results, list):
        return []

    parsed = []

    for item in results:
        if not isinstance(item, dict):
            continue

        classes = item.get("classes") or []

        parsed.append({
            "grid": item.get("grid"),
            "image": item.get("image"),
            "num_boxes": _safe_int(item.get("num_boxes")),
            "classes": classes,
            "class_summary": item.get("class_summary") or _class_summary(classes),
            "total_pests": _safe_int(item.get("total_pests")) or len(classes),
            "avg_conf": _to_float(item.get("avg_conf")),
        })

    return parsed


def create_detection_result(
    user_id,
    filename,
    full_result_image,
    full_summary,
    crop_result_image,
    crop_summary,
):
    db = get_db()
    cursor = db.cursor()
    full_classes = full_summary.get("classes", [])
    crop_classes = crop_summary.get("classes", [])

    try:
        cursor.execute(
            """
            INSERT INTO detection_results (
                user_id,
                filename,
                full_result_image,
                full_classes,
                full_total_pests,
                full_avg_conf,
                full_duration_ms,
                crop_result_image,
                crop_classes,
                crop_total_pests,
                crop_avg_conf,
                crop_duration_ms,
                per_grid_results_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                filename,
                full_result_image,
                _classes_to_text(full_classes),
                _safe_int(full_summary.get("num_boxes")) or len(full_classes),
                full_summary.get("avg_conf", 0.0),
                full_summary.get("duration_ms", 0),
                crop_result_image,
                _classes_to_text(crop_classes),
                _safe_int(crop_summary.get("num_boxes")) or len(crop_classes),
                crop_summary.get("avg_conf", 0.0),
                crop_summary.get("duration_ms", 0),
                _serialize_per_grid_results(
                    crop_summary.get("per_grid_results", [])
                ),
            ),
        )
        db.commit()
        return cursor.lastrowid
    except Exception:
        db.rollback()
        raise
    finally:
        cursor.close()


def get_detection_rows(user_id=None, include_all=False):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    try:
        if include_all:
            cursor.execute(
                """
                SELECT *
                FROM detection_results
                ORDER BY created_at DESC, id DESC
                """
            )
        else:
            cursor.execute(
                """
                SELECT *
                FROM detection_results
                WHERE user_id = %s
                ORDER BY created_at DESC, id DESC
                """,
                (user_id,),
            )

        rows = []

        for row in cursor.fetchall():
            created_at = row.get("created_at")
            full_classes = _classes_from_text(row.get("full_classes"))
            crop_classes = _classes_from_text(row.get("crop_classes"))
            full_total = (
                _safe_int(row.get("full_total_pests")) or len(full_classes)
            )
            crop_total = (
                _safe_int(row.get("crop_total_pests")) or len(crop_classes)
            )

            rows.append({
                "filename": row.get("filename") or "unknown",

                "full_time": _format_dt(created_at),
                "full_img": row.get("full_result_image"),
                "full_classes": _classes_to_text(full_classes) or "-",
                "full_class_summary": _class_summary(full_classes),
                "full_total_pests": full_total,
                "full_avgconf": _to_float(row.get("full_avg_conf")),
                "full_speed": row.get("full_duration_ms") or 0,

                "crop_time": _format_dt(created_at),
                "crop_img": row.get("crop_result_image"),
                "crop_classes": _classes_to_text(crop_classes) or "-",
                "crop_class_summary": _class_summary(crop_classes),
                "crop_total_pests": crop_total,
                "crop_avgconf": _to_float(row.get("crop_avg_conf")),
                "crop_speed": row.get("crop_duration_ms") or 0,
                "per_grid_results": _parse_per_grid_results(
                    row.get("per_grid_results_json")
                ),
            })

        return rows
    finally:
        cursor.close()
