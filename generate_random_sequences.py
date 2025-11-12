#!/usr/bin/env python3
"""
generate_random_sequences.py
================================

可変長モデル用データ収集に利用するランダム文字列を生成するスクリプト。

要件:
- 37キー（a-z, 0-9, space）を全て含む
- 各キーの出現回数が 2 〜 3 回の範囲
- 文字列長はデフォルトで 95 文字
- 再現性を保つため seed 指定が可能
- JSON 形式で結果を保存

Usage:
    python generate_random_sequences.py \
        --count 10 \
        --length 95 \
        --seed 0 \
        --output generated_sequences.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import random

KEYS: Tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyz0123456789 ")
MIN_OCCURRENCES = 2
MAX_OCCURRENCES = 3


@dataclass
class SequenceInfo:
    sequence_id: str
    seed: int
    text: str
    frequencies: Dict[str, int]


def create_occurrence_table(length: int, rng: random.Random) -> Dict[str, int]:
    raise NotImplementedError("create_occurrence_table is no longer used.")


def generate_sequences(
    count: int,
    length: int,
    seed: int,
) -> List[SequenceInfo]:
    if length < MIN_OCCURRENCES * len(KEYS):
        raise ValueError(
            f"length {length} is too short. Minimum length is {MIN_OCCURRENCES * len(KEYS)}"
        )
    if length > MAX_OCCURRENCES * len(KEYS):
        raise ValueError(
            f"length {length} is too long. Maximum length is {MAX_OCCURRENCES * len(KEYS)}"
        )

    rng = random.Random(seed)

    # 各シーケンスの初期出現回数（すべてのキーを2回）
    sequence_occurrences: List[Dict[str, int]] = [
        {key: MIN_OCCURRENCES for key in KEYS} for _ in range(count)
    ]

    # 各シーケンスで追加できる余剰枠（21）
    extra_per_sequence = length - MIN_OCCURRENCES * len(KEYS)
    extras_sequence_remaining: List[int] = [extra_per_sequence for _ in range(count)]

    # データセット全体で割り当てる余剰枠
    total_extra = extra_per_sequence * count
    base_extra_per_key = total_extra // len(KEYS)
    remainder = total_extra % len(KEYS)

    keys_shuffled = list(KEYS)
    rng.shuffle(keys_shuffled)

    extras_per_key = {key: base_extra_per_key for key in KEYS}
    for key in keys_shuffled[:remainder]:
        extras_per_key[key] += 1

    # 余剰枠をキー×シーケンスに割り当てる
    for key in keys_shuffled:
        for _ in range(extras_per_key[key]):
            eligible_sequences = [
                idx
                for idx in range(count)
                if extras_sequence_remaining[idx] > 0
                and sequence_occurrences[idx][key] < MAX_OCCURRENCES
            ]
            if not eligible_sequences:
                raise RuntimeError(
                    f"No eligible sequences left for key '{key}'. "
                    "Try increasing sequence length or max occurrences."
                )

            max_remaining = max(extras_sequence_remaining[idx] for idx in eligible_sequences)
            top_candidates = [
                idx for idx in eligible_sequences if extras_sequence_remaining[idx] == max_remaining
            ]
            seq_idx = rng.choice(top_candidates)

            sequence_occurrences[seq_idx][key] += 1
            extras_sequence_remaining[seq_idx] -= 1

    if any(val != 0 for val in extras_sequence_remaining):
        raise RuntimeError("Failed to distribute all sequence extras evenly.")

    sequences: List[SequenceInfo] = []
    for idx, occurrences in enumerate(sequence_occurrences):
        seq_rng = random.Random(seed + idx)
        pool: List[str] = []
        for key, count_key in occurrences.items():
            pool.extend([key] * count_key)

        if len(pool) != length:
            raise RuntimeError(
                f"Sequence length mismatch: expected {length}, got {len(pool)}"
            )

        seq_rng.shuffle(pool)
        sequence_text = "".join(pool)

        info = SequenceInfo(
            sequence_id=f"sequence_{idx + 1:02d}",
            seed=seed + idx,
            text=sequence_text,
            frequencies=dict(sorted(Counter(sequence_text).items())),
        )
        sequences.append(info)
    return sequences


def save_sequences(
    sequences: List[SequenceInfo], output_path: Path, pretty: bool = True
) -> None:
    result = {
        "metadata": {
            "count": len(sequences),
            "length": len(sequences[0].text) if sequences else 0,
            "keys": "".join(KEYS[:-1]),
            "include_space": True,
            "min_occurrences": MIN_OCCURRENCES,
            "max_occurrences": MAX_OCCURRENCES,
        },
        "sequences": {
            seq.sequence_id: {
                "seed": seq.seed,
                "text": seq.text,
                "frequencies": seq.frequencies,
            }
            for seq in sequences
        },
    }

    with output_path.open("w", encoding="utf-8") as fp:
        if pretty:
            json.dump(result, fp, ensure_ascii=False, indent=2)
        else:
            json.dump(result, fp, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random key sequences for variable-length data collection."
    )
    parser.add_argument("--count", type=int, default=10, help="生成する配列数")
    parser.add_argument("--length", type=int, default=95, help="各配列の文字数")
    parser.add_argument("--seed", type=int, default=0, help="シード値（先頭配列に適用）")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated_sequences.json"),
        help="生成結果の出力ファイル",
    )
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="JSONをインデントなしで出力する",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = generate_sequences(count=args.count, length=args.length, seed=args.seed)
    save_sequences(sequences, args.output, pretty=not args.no_pretty)

    print(f"[OK] Generated {len(sequences)} sequences -> {args.output}")
    for seq in sequences:
        print(
            f"  - {seq.sequence_id}: seed={seq.seed}, "
            f"unique_keys={len(seq.frequencies)}, length={len(seq.text)}"
        )


if __name__ == "__main__":
    main()

