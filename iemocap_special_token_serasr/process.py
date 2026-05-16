#!/usr/bin/env python3
import os
import pandas as pd

CSV_DIR = "path/to/iemocap"

# Directory where the script is located (root output directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SESSIONS = [1, 2, 3, 4, 5]
SPLITS = ["train", "test"]

# OCEAN mapping table: binary label -> descriptive token
TRAIT_WORD_MAP = {
    "O": {"e0": "<openness_low>", "e1": "<openness_high>"},
    "C": {"e0": "<conscientiousness_low>", "e1": "<conscientiousness_high>"},
    "E": {"e0": "<extraversion_low>", "e1": "<extraversion_high>"},
    "A": {"e0": "<agreeableness_low>", "e1": "<agreeableness_high>"},
    "N": {"e0": "<neuroticism_low>", "e1": "<neuroticism_high>"},
}


def bin_to_key(v):
    """
    Convert values in bin_* columns into 'e0' or 'e1':
      - If already 'e0'/'e1' (case-insensitive), return directly
      - If '0' or '1', convert to 'e0'/'e1'
    """
    s = str(v).strip().lower()
    if s in ("e0", "e1"):
        return s
    if s in ("0", "1"):
        return f"e{s}"
    raise ValueError(f"Cannot parse value '{v}' into e0/e1 for personality mapping.")


def process_csv(csv_path, out_dir):
    """Read CSV and generate wav.scp, text, and personality files."""
    print(f"Processing {csv_path} ...")

    df = pd.read_csv(csv_path)

    # Required column check
    required_cols = ["file", "text", "emotion"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{csv_path} is missing required column: {c}")

    # Personality binary columns
    bin_cols = [
        "e",
        "a",
        "c",
        "n",
        "o",
    ]
    for c in bin_cols:
        if c not in df.columns:
            raise ValueError(f"{csv_path} is missing personality binary column: {c}")

    os.makedirs(out_dir, exist_ok=True)

    # uttid: remove extension and keep filename only
    df["uttid"] = df["file"].apply(lambda x: os.path.splitext(os.path.basename(str(x)))[0])

    # emotion token + original text
    df["emotion_lower"] = df["emotion"].astype(str).str.lower()
    df["orig_text"] = df["text"].astype(str).fillna("").str.strip()

    # Target format: <emotion> + space + text
    df["text_out"] = "<" + df["emotion_lower"] + ">" + " " + df["orig_text"]

    # Sort by uttid for stable output
    df = df.sort_values("uttid")

    wavscp_path = os.path.join(out_dir, "wav.scp")
    text_path = os.path.join(out_dir, "text")
    personality_path = os.path.join(out_dir, "personality")

    # Write wav.scp
    with open(wavscp_path, "w", encoding="utf-8") as f_wav:
        for _, r in df.iterrows():
            f_wav.write(f"{r['uttid']} {r['file']}\n")

    # Write text
    with open(text_path, "w", encoding="utf-8") as f_txt:
        for _, r in df.iterrows():
            f_txt.write(f"{r['uttid']} {r['text_out']}\n")

    # Write personality
    # Convert binary labels into descriptive tokens, then uppercase the whole sentence
    with open(personality_path, "w", encoding="utf-8") as f_per:
        for _, r in df.iterrows():
            b_ext = bin_to_key(r["e"])
            b_agr = bin_to_key(r["a"])
            b_con = bin_to_key(r["c"])
            b_neu = bin_to_key(r["n"])
            b_ope = bin_to_key(r["o"])

            w_e = TRAIT_WORD_MAP["E"][b_ext]
            w_a = TRAIT_WORD_MAP["A"][b_agr]
            w_c = TRAIT_WORD_MAP["C"][b_con]
            w_n = TRAIT_WORD_MAP["N"][b_neu]
            w_o = TRAIT_WORD_MAP["O"][b_ope]

            line = (
                "THE PERSONALITY OF SPEAKER IS "
                f"{w_e}, {w_a}, {w_c}, {w_n}, {w_o}."
            )
            line = line.upper()

            f_per.write(f"{r['uttid']} {line}\n")

    print(f"✅ Output completed: {wavscp_path}, {text_path}, {personality_path}\n")


def main():
    for s in SESSIONS:
        for split in SPLITS:
            base_name = f"session{s}.{split}"
            csv_path = os.path.join(CSV_DIR, f"{base_name}.csv")
            out_dir = os.path.join(SCRIPT_DIR, f"session{s}", split)

            if os.path.exists(csv_path):
                process_csv(csv_path, out_dir)
            else:
                print(f"⚠️ Skipping {base_name}: file not found: {csv_path}")

    print("🎯 All processing completed!")


if __name__ == "__main__":
    main()