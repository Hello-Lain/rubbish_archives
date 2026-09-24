from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class PopeDataset(Dataset[dict[str, Any]]):
    """Read POPE JSONL records and resolve image paths without tokenization."""

    def __init__(
        self,
        path: str,
        images_root: str,
        start: int = 0,
        limit: int | None = None,
        validate_images: bool = False,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.images_root = Path(images_root).expanduser().resolve()
        if start < 0:
            raise ValueError("start must be non-negative")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive when provided")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

        rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        end = None if limit is None else start + limit
        selected = rows[start:end]
        if not selected:
            raise ValueError(
                f"No POPE rows selected from {self.path} with start={start}, limit={limit}"
            )
        self._records = [
            self._normalize_row(row, index=start + index) for index, row in enumerate(selected)
        ]
        if validate_images:
            missing = [
                record["image_path"]
                for record in self._records
                if not Path(record["image_path"]).is_file()
            ]
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} POPE images; first={missing[0]}")

    def _normalize_row(self, row: dict[str, Any], index: int) -> dict[str, Any]:
        image_name = row.get("image") or row.get("image_path")
        if image_name is None:
            raise KeyError(f"POPE row {index} has no image field")
        question = row.get("text") or row.get("question")
        if question is None:
            raise KeyError(f"POPE row {index} has no text/question field")
        label = str(row.get("label", "")).strip().lower()
        if label not in {"yes", "no"}:
            raise ValueError(f"POPE row {index} has unsupported label={label!r}")

        image_path = Path(str(image_name))
        if not image_path.is_absolute():
            image_path = self.images_root / image_path
        record = dict(row)
        record.update(
            {
                "id": str(row.get("question_id", index)),
                "question_id": row.get("question_id", index),
                "prompt": str(question),
                "image_path": str(image_path.resolve()),
                "label": label,
            }
        )
        return record

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self._records[index])
