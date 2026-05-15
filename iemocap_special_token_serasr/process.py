#!/usr/bin/env python3
import os
import pandas as pd

# ======== CSV 固定目录 ========
CSV_DIR = "/lustre/users/gao/code/github_repo/SLM_personality/data"

# 脚本所在目录（输出目录的根目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SESSIONS = [1, 2, 3, 4, 5]
SPLITS = ["train", "test"]

# OCEAN 映射表：二值 -> 描述词
TRAIT_WORD_MAP = {
    "O": {"e0": "<openness_low>", "e1": "<openness_high>"},
    "C": {"e0": "<conscientiousness_low>", "e1": "<conscientiousness_high>"},
    "E": {"e0": "<extraversion_low>", "e1": "<extraversion_high>"},
    "A": {"e0": "<agreeableness_low>", "e1": "<agreeableness_high>"},
    "N": {"e0": "<neuroticism_low>", "e1": "<neuroticism_high>"},
}


def bin_to_key(v):
    """
    将 bin_* 列的值转成 'e0' 或 'e1'：
      - 如果本身是 'e0'/'e1'（不区分大小写）直接返回
      - 如果是 '0' 或 '1'，转成 'e0'/'e1'
    """
    s = str(v).strip().lower()
    if s in ("e0", "e1"):
        return s
    if s in ("0", "1"):
        return f"e{s}"
    raise ValueError(f"无法从值 '{v}' 解析出 e0/e1，用于 personality 映射。")


def process_csv(csv_path, out_dir):
    """读取CSV并生成 wav.scp、text、personality（映射后的单词，全大写）"""
    print(f"处理 {csv_path} ...")

    df = pd.read_csv(csv_path)

    # 必要列检查
    required_cols = ["file", "text", "emotion"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{csv_path} 缺少必要列: {c}")

    # personality 的 bin 列
    bin_cols = [
        "e",
        "a",
        "c",
        "n",
        "o",
    ]
    for c in bin_cols:
        if c not in df.columns:
            raise ValueError(f"{csv_path} 缺少性格二值列: {c}")

    os.makedirs(out_dir, exist_ok=True)

    # uttid: 去掉扩展名，只保留文件名
    df["uttid"] = df["file"].apply(lambda x: os.path.splitext(os.path.basename(str(x)))[0])

    # emotion token + 原始 text
    df["emotion_lower"] = df["emotion"].astype(str).str.lower()
    df["orig_text"] = df["text"].astype(str).fillna("").str.strip()
    # 目标：<emotion> + 空格 + text
    df["text_out"] = "<" + df["emotion_lower"] + ">" + " " + df["orig_text"]

    # 按 uttid 排序，保证稳定
    df = df.sort_values("uttid")

    wavscp_path = os.path.join(out_dir, "wav.scp")
    text_path = os.path.join(out_dir, "text")
    personality_path = os.path.join(out_dir, "personality")

    # 写 wav.scp
    with open(wavscp_path, "w", encoding="utf-8") as f_wav:
        for _, r in df.iterrows():
            f_wav.write(f"{r['uttid']} {r['file']}\n")

    # 写 text
    with open(text_path, "w", encoding="utf-8") as f_txt:
        for _, r in df.iterrows():
            f_txt.write(f"{r['uttid']} {r['text_out']}\n")

    # 写 personality（从 bin_* 列映射成单词，然后整行大写）
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

    print(f"✅ 输出完成：{wavscp_path}, {text_path}, {personality_path}\n")


def main():
    for s in SESSIONS:
        for split in SPLITS:
            base_name = f"session{s}.{split}"
            csv_path = os.path.join(CSV_DIR, f"{base_name}.csv")
            out_dir = os.path.join(SCRIPT_DIR, f"session{s}", split)

            if os.path.exists(csv_path):
                process_csv(csv_path, out_dir)
            else:
                print(f"⚠️  跳过 {base_name}：未找到 {csv_path}")

    print("🎯 全部完成！")


if __name__ == "__main__":
    main()
