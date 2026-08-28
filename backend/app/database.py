from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "backend" / "data"

DATABASE_PATH = DATA_DIR / "image_quality.db"


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False,
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database() -> None:
    connection = get_connection()

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                image_name TEXT NOT NULL,

                degradation TEXT NOT NULL,

                severity TEXT NOT NULL,

                quality_score REAL NOT NULL,

                quality_label TEXT NOT NULL,

                issues TEXT NOT NULL,

                image_statistics TEXT NOT NULL,

                degradation_confidence REAL NOT NULL,

                severity_confidence REAL NOT NULL,

                degradation_probabilities TEXT NOT NULL,

                severity_probabilities TEXT NOT NULL,

                created_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.commit()

        migrate_database(connection)

    finally:
        connection.close()


def migrate_database(
    connection: sqlite3.Connection,
) -> None:
    columns = connection.execute(
        """
        PRAGMA table_info(analyses)
        """
    ).fetchall()

    existing_columns = {
        column["name"]
        for column in columns
    }

    migrations = {
        "quality_label": (
            "ALTER TABLE analyses "
            "ADD COLUMN quality_label "
            "TEXT NOT NULL DEFAULT 'DEGRADED'"
        ),
        "issues": (
            "ALTER TABLE analyses "
            "ADD COLUMN issues "
            "TEXT NOT NULL DEFAULT '[]'"
        ),
        "image_statistics": (
            "ALTER TABLE analyses "
            "ADD COLUMN image_statistics "
            "TEXT NOT NULL DEFAULT '{}'"
        ),
    }

    for column_name, statement in migrations.items():
        if column_name not in existing_columns:
            connection.execute(statement)

    connection.commit()


def serialize_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
    )


def deserialize_json(
    value: str | None,
    default: Any,
) -> Any:
    if not value:
        return default

    try:
        return json.loads(value)

    except (
        json.JSONDecodeError,
        TypeError,
    ):
        return default


def save_analysis(
    result: dict[str, Any],
) -> int:

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            INSERT INTO analyses (
                image_name,
                degradation,
                severity,
                quality_score,
                quality_label,
                issues,
                image_statistics,
                degradation_confidence,
                severity_confidence,
                degradation_probabilities,
                severity_probabilities
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                result["image"],
                result["degradation"],
                result["severity"],
                result["quality_score"],
                result["quality_label"],
                serialize_json(
                    result["issues"]
                ),
                serialize_json(
                    result["image_statistics"]
                ),
                result[
                    "degradation_confidence"
                ],
                result[
                    "severity_confidence"
                ],
                serialize_json(
                    result[
                        "degradation_probabilities"
                    ]
                ),
                serialize_json(
                    result[
                        "severity_probabilities"
                    ]
                ),
            ),
        )

        connection.commit()

        return int(
            cursor.lastrowid
        )

    finally:
        connection.close()


def row_to_analysis(
    row: sqlite3.Row,
) -> dict[str, Any]:

    return {
        "id": row["id"],

        "image": row["image_name"],

        "degradation": row["degradation"],

        "severity": row["severity"],

        "quality_score": row[
            "quality_score"
        ],

        "quality_label": row[
            "quality_label"
        ],

        "issues": deserialize_json(
            row["issues"],
            [],
        ),

        "image_statistics": deserialize_json(
            row["image_statistics"],
            {},
        ),

        "degradation_confidence": row[
            "degradation_confidence"
        ],

        "severity_confidence": row[
            "severity_confidence"
        ],

        "degradation_probabilities":
            deserialize_json(
                row[
                    "degradation_probabilities"
                ],
                {},
            ),

        "severity_probabilities":
            deserialize_json(
                row[
                    "severity_probabilities"
                ],
                {},
            ),

        "created_at": row[
            "created_at"
        ],
    }


def get_analysis_history(
    limit: int = 20,
) -> list[dict[str, Any]]:

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                id,
                image_name,
                degradation,
                severity,
                quality_score,
                quality_label,
                issues,
                image_statistics,
                degradation_confidence,
                severity_confidence,
                degradation_probabilities,
                severity_probabilities,
                created_at
            FROM analyses
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [
            row_to_analysis(row)
            for row in rows
        ]

    finally:
        connection.close()


def get_analysis(
    analysis_id: int,
) -> dict[str, Any] | None:

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                id,
                image_name,
                degradation,
                severity,
                quality_score,
                quality_label,
                issues,
                image_statistics,
                degradation_confidence,
                severity_confidence,
                degradation_probabilities,
                severity_probabilities,
                created_at
            FROM analyses
            WHERE id = ?
            """,
            (analysis_id,),
        ).fetchone()

        if row is None:
            return None

        return row_to_analysis(row)

    finally:
        connection.close()