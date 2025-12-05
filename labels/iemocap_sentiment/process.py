#!/usr/bin/env python3
import os
import pandas as pd

# ======== CSV 固定目录 ========
CSV_DIR = "/lustre/users/gao/speechllm/labels/iemocap_labels_all"

# 脚本所在目录（输出目录的根目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SESSIONS = [1, 2, 3, 4, 5]
SPLITS = ["train", "test"]


def process_csv(csv_path, out_dir):
    """读取CSV并生成 wav.scp 和 text（emotion 小写标签）"""
    print(f"处理 {csv_path} ...")

    df = pd.read_csv(csv_path)
    required_cols = ["file", "text", "emotion"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{csv_path} 缺少必要列: {c}")

    os.makedirs(out_dir, exist_ok=True)

    # uttid: 去掉扩展名，只保留文件名
    df["uttid"] = df["file"].apply(
        lambda x: os.path.splitext(os.path.basename(str(x)))[0]
    )

    # emotion 小写 + 拼接 text_out: "<emotion> 文本"
    df["emotion_lower"] = df["sentiment"].astype(str).str.lower()
    df["text_out"] = "<" + df["emotion_lower"] + ">"

    # 排序保持稳定
    df = df.sort_values("uttid")

    wavscp_path = os.path.join(out_dir, "wav.scp")
    text_path = os.path.join(out_dir, "text")

    # 写 wav.scp
    with open(wavscp_path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(f"{r['uttid']} {r['file']}\n")

    # 写 text
    with open(text_path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(f"{r['uttid']} {r['text_out']}\n")

    print(f"✅ 输出完成：{wavscp_path}, {text_path}\n")


def main():
    for s in SESSIONS:
        for split in SPLITS:
            base_name = f"session{s}.{split}"
            csv_path = os.path.join(CSV_DIR, f"{base_name}.csv")
            # 输出结构：SCRIPT_DIR/session{s}/{split}/
            out_dir = os.path.join(SCRIPT_DIR, f"session{s}", split)

            if os.path.exists(csv_path):
                process_csv(csv_path, out_dir)
            else:
                print(f"⚠️  跳过 {base_name}：未找到 {csv_path}")

    print("🎯 全部完成！")


if __name__ == "__main__":
    main()
