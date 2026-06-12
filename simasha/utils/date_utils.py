from datetime import datetime, timedelta


def parse_iso(s):
    if not s:
        return None

    try:
        if s.endswith("Z"):
            s = s[:-1]

        return datetime.fromisoformat(s)

    except Exception:
        return None


def fmt_local(iso_str):
    dt = parse_iso(iso_str)

    if not dt:
        return "-"

    dt = dt + timedelta(hours=7)

    return dt.strftime("%Y-%m-%d %H:%M:%S")
