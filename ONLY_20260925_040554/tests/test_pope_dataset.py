from __future__ import annotations

import json
from pathlib import Path

from data.pope import PopeDataset


def test_pope_dataset_only_normalizes_raw_records(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "0001.jpg"
    image_path.write_bytes(b"not decoded by the dataset")
    pope_path = tmp_path / "pope.jsonl"
    pope_path.write_text(
        json.dumps(
            {
                "question_id": 17,
                "image": "0001.jpg",
                "text": "Is there a cat?",
                "label": "yes",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = PopeDataset(
        path=str(pope_path),
        images_root=str(image_root),
        validate_images=True,
    )
    record = dataset[0]

    assert record["id"] == "17"
    assert record["prompt"] == "Is there a cat?"
    assert record["label"] == "yes"
    assert record["image_path"] == str(image_path.resolve())
