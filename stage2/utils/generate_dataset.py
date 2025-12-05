#!/usr/bin/env python3
"""
Created by Yuan at 2025-12-03

generate_dataset.py – Build a Hugging Face Dataset from espnet-style session splits.

Input directory structure (base_data_path):

    base_data_path/
        session1/
            train/
                wav.scp
                text
                personality
            test/
                wav.scp
                text
                personality
        ...
        session5/
            train/
                wav.scp
                text
                personality
            test/
                wav.scp
                text
                personality

Output directory structure (output_dir, default: derived from base_data_path):

    output_dir/
        session1/
            train/
                # HF dataset (DatasetDict.save_to_disk)
            test/
                # HF dataset
        ...
        session5/
            train/
            test/
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict

import pandas as pd
from datasets import Audio, Dataset, DatasetDict

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build a Hugging Face Dataset from session-wise espnet-style files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base_data_path",
        type=str,
        default="/lustre/users/gao/speechllm/labels/iemocap",
        help="Root directory that contains sessionX/train and sessionX/test folders.",
    )
    parser.add_argument(
        "--wav_scp_name",
        type=str,
        default="wav.scp",
        help="Name of the SCP file (e.g. `wav.scp`, `wav_clean.scp`).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Root output directory for HF datasets. "
            "Each session will be saved to output_dir/sessionX/ as a DatasetDict(train,test)."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Based on the speech, identify the speaker's current emotion."
        ),
        help=(
            "Task description used under the [Task] block. "
            "Final prompt format:\n"
            "[Speaker Personality]\\n"
            "${PERSONALITY_BLOCK}\\n\\n"
            "[Task]\\n"
            "<this prompt text>"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def build_file_paths(
    *,
    base: str,
    wav_name: str,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Construct wav.scp, text & personality paths for each session + split.

    We assume under `base`:

        session1/train, session1/test,
        ...
        session5/train, session5/test

    Return format:
        {
            "session1": {
                "train": {"wav_scp": "...", "text": "...", "personality": "..."},
                "test":  {"wav_scp": "...", "text": "...", "personality": "..."},
            },
            ...
        }
    """

    paths: Dict[str, Dict[str, Dict[str, str]]] = {}

    for sess_id in range(1, 6):
        session_name = f"session{sess_id}"
        session_dir = os.path.join(base, session_name)

        paths[session_name] = {}
        for subset in ("train", "test"):
            subset_dir = os.path.join(session_dir, subset)
            wav_scp_path = os.path.join(subset_dir, wav_name)
            text_path = os.path.join(subset_dir, "text")
            personality_path = os.path.join(subset_dir, "personality")

            paths[session_name][subset] = {
                "wav_scp": wav_scp_path,
                "text": text_path,
                "personality": personality_path,
            }

            LOGGER.info("[%s.%s] wav_scp      → %s", session_name, subset, wav_scp_path)
            LOGGER.info("[%s.%s] text         → %s", session_name, subset, text_path)
            LOGGER.info("[%s.%s] personality  → %s", session_name, subset, personality_path)

    return paths


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def parse_line(line: str):
    parts = line.strip().split(" ", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def process_split(
    wav_scp_path: str,
    text_path: str,
    personality_path: str,
    prompt: str,
) -> Dataset:
    """Convert a trio of wav.scp, text & personality files into a HF *Dataset*."""

    # wav.scp → (id, path)
    with open(wav_scp_path, "r") as f:
        audio_df = pd.DataFrame([parse_line(l) for l in f], columns=["id", "path"])

    # text → (id, text)
    with open(text_path, "r") as f:
        text_df = pd.DataFrame([parse_line(l) for l in f], columns=["id", "text"])

    # personality → (id, personality_text)
    # example line:
    # Ses02F_impro01_F000 THE PERSONALITY OF SPEAKER IS EXTRAVERSION 5.2,AGREEABLENESS 2.4,...
    with open(personality_path, "r") as f:
        personality_df = pd.DataFrame(
            [parse_line(l) for l in f],
            columns=["id", "personality"],
        )

    # Merge by id
    merged = (
        audio_df
        .merge(text_df, on="id", how="inner")
        .merge(personality_df, on="id", how="left")
    )

    # Build samples:
    # [Speaker Personality]
    # ${PERSONALITY_BLOCK}
    #
    # [Task]
    # <prompt>
    def build_prompt(row):
        personality_block = ""
        if isinstance(row.personality, str):
            personality_block = row.personality.strip()

        if personality_block:
            return (
                "[Speaker Personality]\n"
                f"{personality_block}\n\n"
                "[Task]\n"
                f"{prompt}"
            )
        else:
            # If personality is missing or empty, only provide the [Task] block
            return (
                "[Task]\n"
                f"{prompt}"
            )

    ds = Dataset.from_list(
        [
            {
                "id": row.id,
                "audio": row.path,
                "text": row.text,
                "prompt": build_prompt(row),
            }
            for row in merged.itertuples(index=False)
        ]
    )

    return ds.cast_column("audio", Audio(sampling_rate=16_000))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    # Decide root output directory
    if args.output_dir is not None:
        output_root = args.output_dir
    else:
        base_name = os.path.basename(os.path.normpath(args.base_data_path))
        # By default, create a directory with the same name (or "hf_dataset") in the current directory
        output_root = base_name or "hf_dataset"

    os.makedirs(output_root, exist_ok=True)

    # Resolve all paths
    file_paths = build_file_paths(
        base=args.base_data_path,
        wav_name=args.wav_scp_name,
    )

    # For each session, build a DatasetDict(train, test) and save to output_root/sessionX
    for session_name, splits in file_paths.items():
        logging.info("Processing %s ...", session_name)

        # Build each split separately and print the last sample's prompt after each split
        ds_splits: Dict[str, Dataset] = {}
        for split, paths in splits.items():
            ds_split = process_split(
                paths["wav_scp"],
                paths["text"],
                paths["personality"],
                args.prompt,
            )
            ds_splits[split] = ds_split

            # Print the last sample's prompt of the current split
            if len(ds_split) > 0:
                last_prompt = ds_split[-1]["prompt"]
                logging.info(
                    "Last prompt of %s/%s:\n%s",
                    session_name,
                    split,
                    last_prompt,
                )
            else:
                logging.warning("Dataset %s/%s is empty.", session_name, split)

        ds_session = DatasetDict(ds_splits)

        out_dir_session = os.path.join(output_root, session_name)
        ds_session.save_to_disk(out_dir_session)

        logging.info("Dataset for %s saved to %s", session_name, out_dir_session)
        logging.info(ds_session)

        first_split = next(iter(ds_session.keys()))
        logging.info(
            "First sample of %s/%s: %s",
            session_name,
            first_split,
            ds_session[first_split][0],
        )


if __name__ == "__main__":
    main()
