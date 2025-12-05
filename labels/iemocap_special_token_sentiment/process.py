#!/usr/bin/env python3
import os
import pandas as pd

# ======== Fixed CSV directory ========
CSV_DIR = "/lustre/users/gao/speechllm/labels/iemocap_labels_all"

# Script directory (root for output directories)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SESSIONS = [1, 2, 3, 4, 5]
SPLITS = ["train", "test"]

# OCEAN mapping table: binary -> special token
TRAIT_WORD_MAP = {
    "O": {"e0": "<openness_low>",          "e1": "<openness_high>"},
    "C": {"e0": "<conscientiousness_low>", "e1": "<conscientiousness_high>"},
    "E": {"e0": "<extraversion_low>",      "e1": "<extraversion_high>"},
    "A": {"e0": "<agreeableness_low>",     "e1": "<agreeableness_high>"},
    "N": {"e0": "<neuroticism_low>",       "e1": "<neuroticism_high>"},
}


def bin_to_key(v):
    """
    Convert value of bin_* columns into 'e0' or 'e1':
      - If the value is already 'e0'/'e1' (case-insensitive), return it.
      - If the value is '0' or '1', convert to 'e0'/'e1'.
    """
    s = str(v).strip().lower()
    if s in ("e0", "e1"):
        return s
    if s in ("0", "1"):
        return f"e{s}"
    raise ValueError(
        f"Cannot parse value '{v}' to e0/e1 for personality mapping."
    )


def process_csv(csv_path, out_dir):
    """Read CSV and generate wav.scp, text, and personality files."""
    print(f"Processing {csv_path} ...")

    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = ["file", "text", "sentiment"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{csv_path} is missing required column: {c}")

    # Personality binary columns
    bin_cols = [
        "bin_Extraversion",
        "bin_Agreeableness",
        "bin_Conscientiousness",
        "bin_Neuroticism",
        "bin_Openness",
    ]
    for c in bin_cols:
        if c not in df.columns:
            raise ValueError(f"{csv_path} is missing personality binary column: {c}")

    os.makedirs(out_dir, exist_ok=True)

    # uttid: strip extension, keep only filename
    df["uttid"] = df["file"].apply(
        lambda x: os.path.splitext(os.path.basename(str(x)))[0]
    )

    # emotion token
    df["emotion_lower"] = df["sentiment"].astype(str).str.lower()
    df["text_out"] = "<" + df["emotion_lower"] + ">"

    # Sort by uttid to keep output stable
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

    # Write personality (map bin_* columns to tokens, then uppercase the sentence)
    with open(personality_path, "w", encoding="utf-8") as f_per:
        for _, r in df.iterrows():
            # Read E, A, C, N, O binary values
            b_ext = bin_to_key(r["bin_Extraversion"])
            b_agr = bin_to_key(r["bin_Agreeableness"])
            b_con = bin_to_key(r["bin_Conscientiousness"])
            b_neu = bin_to_key(r["bin_Neuroticism"])
            b_ope = bin_to_key(r["bin_Openness"])

            # Map to trait tokens
            w_e = TRAIT_WORD_MAP["E"][b_ext]
            w_a = TRAIT_WORD_MAP["A"][b_agr]
            w_c = TRAIT_WORD_MAP["C"][b_con]
            w_n = TRAIT_WORD_MAP["N"][b_neu]
            w_o = TRAIT_WORD_MAP["O"][b_ope]

            # Use tokens instead of numbers; keep order: E, A, C, N, O
            line = (
                "THE PERSONALITY OF SPEAKER IS "
                f"{w_e}, {w_a}, {w_c}, {w_n}, {w_o}."
            )

            # Uppercase the whole sentence (tokens are already uppercase-like)
            line = line.upper()

            f_per.write(f"{r['uttid']} {line}\n")

    print(f"✅ Finished: {wavscp_path}, {text_path}, {personality_path}\n")


def main():
    for s in SESSIONS:
        for split in SPLITS:
            base_name = f"session{s}.{split}"
            csv_path = os.path.join(CSV_DIR, f"{base_name}.csv")
            out_dir = os.path.join(SCRIPT_DIR, f"session{s}", split)

            if os.path.exists(csv_path):
                process_csv(csv_path, out_dir)
            else:
                print(f"⚠️  Skip {base_name}: {csv_path} not found")

    print("🎯 All done!")


if __name__ == "__main__":
    main()
