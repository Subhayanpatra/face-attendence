import csv
from datetime import datetime
import os

FILE_PATH = "data/attendance.csv"

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")

    rows = []
    last_row = None

    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for r in reversed(rows):
                if r["Name"] == name and r["Date"] == today:
                    last_row = r
                    break

    # ------------------ DECISION ------------------
    if last_row is None or last_row["Punch Out"]:
        # Punch IN
        rows.append({
            "Name": name,
            "Date": today,
            "Punch In": now,
            "Punch Out": ""
        })
        action = "Punch In"

    else:
        # Punch OUT
        last_row["Punch Out"] = now
        action = "Punch Out"

    # ------------------ WRITE BACK ------------------
    with open(FILE_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Name", "Date", "Punch In", "Punch Out"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return action