"""
generate_data.py
-----------------
Creates the synthetic cybersecurity log dataset used throughout the project.

This is a STANDALONE, one-time script (not part of the live app). Run it
whenever you want to regenerate data/synthetic_security_logs.csv:

    python generate_data.py

Design notes
------------
- Every record belongs to one of a fixed pool of synthetic users, each with
  a "normal" home login location and typical behaviour.
- ~78% of records are generated to look like normal, everyday activity.
- ~22% of records are generated to look suspicious (the kind a SOC analyst
  or an anomaly-detection model should flag) — e.g. many failed logins,
  logins from a location far from the user's usual one, unusually large
  data transfers, scans against many ports/hosts, or activity at odd hours.
- Suspicion is expressed as deviation from a user's OWN normal baseline
  (impossible travel, spikes vs. their usual volume) rather than by
  labelling any specific country/region as inherently suspicious — this
  keeps the dataset realistic and avoids geographic bias.
- An extra `true_label` column ("normal" / "suspicious") is included ONLY
  so the student can later validate/evaluate the anomaly detection and
  rule engine. It is ground truth used for testing — later pipeline
  phases should treat the rest of the columns as the real input and use
  this column only to check their own results, not as a model feature.

This is purely defensive: it simulates monitoring/log data for a security
dashboard. No attacks are executed and no offensive techniques are used.
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import COMMON_PORTS, LOG_FILE_PATH, NUM_RECORDS, RANDOM_SEED, SUSPICIOUS_RATIO

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------
# Reference pools
# ---------------------------------------------------------------------
USERS = [
    "asharma", "rpatel", "jkhan", "svc_backup", "mrao", "dgupta",
    "svc_monitoring", "tverma", "nlal", "pkumar", "aiyer", "rsingh",
    "svc_deploy", "kjoshi", "ymehta", "bshah", "cfernandes", "hnair",
    "greddy", "admin_ops",
]

# Each user has one usual home location -> deviations from it look suspicious
HOME_LOCATIONS = [
    "Mumbai, IN", "Bangalore, IN", "Pune, IN", "Delhi, IN", "Hyderabad, IN",
    "Chennai, IN", "London, UK", "Singapore, SG", "New York, US",
    "Toronto, CA", "Sydney, AU", "Dublin, IE",
]
USER_HOME_LOCATION = {u: random.choice(HOME_LOCATIONS) for u in USERS}

# A separate pool of "far away / unusual" locations used to simulate
# impossible-travel style anomalies (kept diverse and non-stereotyping)
UNUSUAL_LOCATIONS = [
    "Lagos, NG", "Sao Paulo, BR", "Bucharest, RO", "Jakarta, ID",
    "Cairo, EG", "Reykjavik, IS", "Nairobi, KE", "Vladivostok, RU",
    "Manila, PH", "Warsaw, PL", "Auckland, NZ", "Baku, AZ",
]

EVENT_TYPES_NORMAL = [
    "login_success", "file_access", "email_activity",
    "vpn_connection", "database_query", "normal_browsing",
]
EVENT_TYPES_SUSPICIOUS = [
    "login_failure", "brute_force_attempt", "network_scan_detected",
    "unusual_login_location", "large_data_transfer", "privilege_escalation_attempt",
    "off_hours_access",
]

# COMMON_PORTS now lives in config.py - it's the same "known common ports"
# knowledge the Phase 3 rule engine uses for its unusual_port fact, so both
# stay in sync from one source of truth.
UNUSUAL_PORTS = [4444, 6667, 31337, 12345, 8081, 1337, 9001]

INTERNAL_SUBNET = "10.0.{}.{}"
SERVER_SUBNET = "10.0.5.{}"     # internal application/db servers
EXTERNAL_IP_POOLS = [
    "203.0.113.{}", "198.51.100.{}", "192.0.2.{}", "45.33.{}.{}",
]


def random_internal_ip():
    return INTERNAL_SUBNET.format(random.randint(1, 20), random.randint(2, 254))


def random_server_ip():
    return SERVER_SUBNET.format(random.randint(2, 60))


def random_external_ip():
    pool = random.choice(EXTERNAL_IP_POOLS)
    parts = pool.count("{}")
    if parts == 1:
        return pool.format(random.randint(2, 254))
    return pool.format(random.randint(1, 254), random.randint(2, 254))


def random_timestamp(days_back=30, off_hours=False):
    now = datetime(2026, 9, 17, 12, 0, 0)
    day_offset = random.randint(0, days_back)
    base_day = now - timedelta(days=day_offset)
    if off_hours:
        hour = random.choice([0, 1, 2, 3, 4, 23])
    else:
        hour = random.randint(8, 20)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return base_day.replace(hour=hour, minute=minute, second=second)


def make_normal_record():
    user = random.choice(USERS)
    return {
        "timestamp": random_timestamp(off_hours=False),
        "user": user,
        "source_ip": random_internal_ip(),
        "destination_ip": random_server_ip(),
        "failed_logins": random.choice([0, 0, 0, 1, 1, 2]),
        "request_count": random.randint(1, 45),
        "port": random.choice(COMMON_PORTS),
        "data_transferred_mb": round(np.random.exponential(scale=8) + 0.1, 2),
        "session_duration": random.randint(30, 3600),
        "event_type": random.choice(EVENT_TYPES_NORMAL),
        "login_location": USER_HOME_LOCATION[user],
        "device_count": random.choice([1, 1, 1, 2]),
        "true_label": "normal",
    }


def make_suspicious_record():
    user = random.choice(USERS)
    pattern = random.choice([
        "brute_force", "impossible_travel", "data_exfiltration",
        "port_scan", "off_hours_privilege", "many_devices",
    ])

    record = {
        "timestamp": random_timestamp(off_hours=True if pattern in
                                       ("off_hours_privilege", "port_scan") else False),
        "user": user,
        "source_ip": random_external_ip() if pattern in
                     ("impossible_travel", "port_scan", "brute_force") else random_internal_ip(),
        "destination_ip": random_server_ip(),
        "failed_logins": random.randint(0, 2),
        "request_count": random.randint(1, 45),
        "port": random.choice(COMMON_PORTS),
        "data_transferred_mb": round(np.random.exponential(scale=8) + 0.1, 2),
        "session_duration": random.randint(30, 3600),
        "event_type": "login_success",
        "login_location": USER_HOME_LOCATION[user],
        "device_count": random.choice([1, 1, 2]),
        "true_label": "suspicious",
    }

    if pattern == "brute_force":
        record.update({
            "failed_logins": random.randint(6, 20),
            "event_type": "brute_force_attempt",
            "request_count": random.randint(20, 120),
        })
    elif pattern == "impossible_travel":
        record.update({
            "login_location": random.choice(UNUSUAL_LOCATIONS),
            "event_type": "unusual_login_location",
            "failed_logins": random.randint(0, 3),
        })
    elif pattern == "data_exfiltration":
        record.update({
            "data_transferred_mb": round(random.uniform(500, 5000), 2),
            "event_type": "large_data_transfer",
            "session_duration": random.randint(600, 7200),
        })
    elif pattern == "port_scan":
        record.update({
            "port": random.choice(UNUSUAL_PORTS),
            "request_count": random.randint(200, 2000),
            "session_duration": random.randint(5, 120),
            "event_type": "network_scan_detected",
        })
    elif pattern == "off_hours_privilege":
        record.update({
            "event_type": "privilege_escalation_attempt",
            "failed_logins": random.randint(1, 5),
        })
    elif pattern == "many_devices":
        record.update({
            "device_count": random.randint(4, 9),
            "event_type": "login_failure",
            "failed_logins": random.randint(2, 8),
        })

    return record


def generate_dataset(num_records=NUM_RECORDS, suspicious_ratio=SUSPICIOUS_RATIO):
    num_suspicious = int(num_records * suspicious_ratio)
    num_normal = num_records - num_suspicious

    records = [make_normal_record() for _ in range(num_normal)]
    records += [make_suspicious_record() for _ in range(num_suspicious)]

    random.shuffle(records)

    df = pd.DataFrame(records)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv(LOG_FILE_PATH, index=False)
    print(f"Generated {len(df)} records -> {LOG_FILE_PATH}")
    print(df["true_label"].value_counts())
