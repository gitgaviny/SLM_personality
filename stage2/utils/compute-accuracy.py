#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from collections import defaultdict

MISS_TAG = "__MISS__"  # 缺失预测的占位符列名

def read_kv(path):
    d = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                d[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return d

def main():
    if len(sys.argv) != 3:
        print("Usage: compute-accuracy.py GOLD_PATH PRED_PATH", file=sys.stderr)
        print("WA=0.0000\tUA=0.0000")
        sys.exit(1)

    gold = read_kv(sys.argv[1])
    pred = read_kv(sys.argv[2])

    if not gold:
        print("WA=0.0000\tUA=0.0000")
        sys.exit(1)

    # —— 统计总体与分类别 —— #
    total = len(gold)
    correct = 0
    per_tot = defaultdict(int)   # gold中每类的样本数
    per_cor = defaultdict(int)   # gold中每类被正确预测的样本数

    # —— 混淆矩阵：行=gold，列=pred（缺失放在 MISS_TAG 列） —— #
    conf = defaultdict(lambda: defaultdict(int))

    # —— 逐条输出 —— #
    print("## per-sample (ID\tGOLD\tPRED)")
    for ex_id, g in gold.items():
        per_tot[g] += 1
        if ex_id in pred:
            p = pred[ex_id]
            if p == g:
                correct += 1
                per_cor[g] += 1
            conf[g][p] += 1
            print(f"{ex_id}\t{g}\t{p}")
        else:
            conf[g][MISS_TAG] += 1
            print(f"{ex_id}\t{g}\t{MISS_TAG}")

    # —— 计算 WA / UA —— #
    wa = correct / total
    recalls = [(per_cor[c] / per_tot[c]) for c in per_tot]
    ua = sum(recalls) / len(recalls) if recalls else 0.0

    print(f"\nWA={wa:.4f}\tUA={ua:.4f}\n")

    # —— 输出混淆矩阵（行=GOLD，列=PRED） —— #
    # 列集合：所有出现过的预测类（包含 MISS_TAG）
    pred_classes = set()
    for g in conf:
        pred_classes.update(conf[g].keys())

    # 将真正的类别（int）排序；MISS 列（若存在）放在最后
    int_cols = sorted([c for c in pred_classes if c != MISS_TAG])
    cols = int_cols + ([MISS_TAG] if MISS_TAG in pred_classes else [])

    # 打印表头
    header = ["G\\P"] + [str(c) for c in cols]
    print("## confusion matrix (rows=GOLD, cols=PRED)")
    print("\t".join(header))

    # 行顺序：gold 中出现过的类别升序
    gold_rows = sorted(per_tot.keys())
    for g in gold_rows:
        row = [str(g)]
        for c in cols:
            row.append(str(conf[g].get(c, 0)))
        print("\t".join(row))

if __name__ == "__main__":
    main()
