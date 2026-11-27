"""Repair the gap ledger's table structure without rewriting any record.

One-shot repair of the 2026-11-26 state, kept in the repository because it is the
record of exactly what changed and because its content-preservation check is the
argument that nothing was lost.  It is not a general tool: the id range it asserts
is the range that existed when the ledger was broken.

The ledger grew by appending single rows separated by blank lines, and a Markdown
table ends at the first blank line, so most of it rendered as bare `|` text.  This
script rebuilds the structure only:

  * closed rows are collected into one `## 已关` section, ascending by id, with a
    header + delimiter row;
  * the three records that were fused into one physical line are split again;
  * two cells that contain a literal `|` from a code sample get the GFM escape;
  * rows that never had a fourth cell get an empty one (nothing is invented).

Every record's text is compared before/after: the only permitted differences are
the escape (which normalises back to the same text), the empty cells, and the one
cell that was lost when the row was truncated.

Usage:
    python3 .github/project/qa/prep/fix_gaps_tables.py            # dry run, prints the plan
    python3 .github/project/qa/prep/fix_gaps_tables.py --apply    # write
"""

from __future__ import annotations

import pathlib
import re
import sys

LEDGER = pathlib.Path("workspace/audit/gaps.md")
UNSOLVED_HEADING = "## 未证 / 明确没做（按影响排序）"
CLOSED_HEADING = "## 已关（按编号升序；原因与结论写在同一行）"
CLOSED_HEADER = "| # | 缺口 | 原因 | 结论 / 退出条件 |"
CLOSED_DELIM = "|---|---|---|---|"
HANDOFF_HEADING = "## 交接注记"

# The record whose fourth cell was lost when its line was truncated.  The text is
# what actually happened, and it says so.
G67_CELL = (
    "已关（P2.2 的 x86 列）。**本条修复时第 4 格已缺失**（原行在写盘时被截断，句子停在 "
    "「…与 `vdot`」），按事实补全：`x86` 的 int8 点积落地；`arm_neon`/`rvv` 的 `fc/int8` "
    "在同一轮的后续格里完成，见 G-79。"
)

# Cell text that contains a literal `|` from a code sample.  GFM escapes it, and the
# escape normalises back to the same character, so the record's text is unchanged.
ESCAPES = (
    (
        "`quant == NULL || kind == NONE || !(multiplier > 0.0F)`",
        r"`quant == NULL \|\| kind == NONE \|\| !(multiplier > 0.0F)`",
    ),
    ("`git ls-files workspace | wc -l`", r"`git ls-files workspace \| wc -l`"),
)


def cells(row: str) -> list[str]:
    """Split one table row on unescaped pipes, dropping the empty ends."""
    parts = re.split(r"(?<!\\)\|", row)
    return [p.strip() for p in parts[1:-1]]


def gap_id(row: str) -> str:
    return cells(row)[0]


def main() -> int:
    text = LEDGER.read_text(encoding="utf-8")
    lines = text.split("\n")

    def find(pred, what):
        for i, line in enumerate(lines):
            if pred(line):
                return i
        raise SystemExit(f"anchor not found: {what}")

    i_first_closed = find(lambda l: l.startswith("| G-60 |"), "first closed row (G-60)")
    i_unsolved = find(lambda l: l == UNSOLVED_HEADING, "unsolved heading")
    i_rest = find(lambda l: l.startswith("| G-35 |"), "the fused G-35 row")
    i_handoff = find(lambda l: l == HANDOFF_HEADING, "handoff heading")

    pre = lines[:i_first_closed]
    unsolved = lines[i_unsolved:i_rest]
    post = lines[i_handoff:]

    closed_raw = [l for l in lines[i_first_closed:i_unsolved] if l.startswith("|")]
    closed_raw += [l for l in lines[i_rest:i_handoff] if l.startswith("|")]

    # The fused row holds three records: split it back apart.  Detect it by counting
    # gap ids, not pipes -- a cell with a code sample contains pipes too.
    rows: list[str] = []
    fused = 0
    for row in closed_raw:
        marks = [m.start() for m in re.finditer(r"\|\s*G-\d+\s*\|", row)]
        if len(marks) > 1:
            pieces = [row[a:b] for a, b in zip(marks, marks[1:] + [len(row)])]
            for piece in pieces:
                assert len(cells(piece)) == 4, (gap_id(piece), len(cells(piece)))
                rows.append(piece.strip())
            fused += 1
        else:
            rows.append(row.rstrip())

    # Normalise: escape the code-sample pipes, collapse a row that ended up with five
    # cells (two conclusion cells written as two cells), pad the rows that never had a
    # fourth cell, restore the lost cell, and order by id.
    fixed: list[str] = []
    padded: list[str] = []
    merged: list[str] = []

    def escape(row: str) -> str:
        for old, new in ESCAPES:
            row = row.replace(old, new)
        return row

    for row in rows:
        row = escape(row)
        held = cells(row)
        if len(held) == 5:
            row = "| " + " | ".join(held[:3] + [" ".join(held[3:])]) + " |"
            merged.append(gap_id(row))
        elif len(held) == 3:
            # These rows never carried a separate conclusion cell; give it an empty one
            # rather than inventing a sentence.
            row = "| " + " | ".join(held + [""]) + " |"
            padded.append(gap_id(row))
        assert len(cells(row)) == 4, (gap_id(row), len(cells(row)))
        fixed.append(row)

    by_id = {gap_id(r): r for r in fixed}
    assert len(by_id) == len(fixed), "a gap id appears twice"
    ids = sorted(int(k[2:]) for k in by_id)
    assert ids == list(range(35, 79)), (ids[0], ids[-1], len(ids), sorted(set(range(35, 79)) - set(ids)))
    by_id["G-67"] = by_id["G-67"].rstrip()
    if cells(by_id["G-67"])[3] == "":
        by_id["G-67"] = re.sub(r"\|\s*\|$", "| " + G67_CELL + " |", by_id["G-67"])
    assert cells(by_id["G-67"])[3], "G-67 still has no conclusion cell"

    new_closed = [by_id[f"G-{n}"] for n in range(35, 79)]

    # ---- content preservation, measured not assumed -----------------------------
    def blobs(rows_in: list[str]) -> dict[str, str]:
        out = {}
        for r in rows_in:
            text = "".join(cells(r)).replace("\\|", "|")
            out.setdefault(gap_id(r), "")
            out[gap_id(r)] += re.sub(r"\s+", "", text)
        return out

    old_blobs = blobs([escape(r) for r in rows])
    new_blobs = blobs(new_closed)
    assert set(old_blobs) == set(new_blobs), (set(old_blobs) ^ set(new_blobs))
    for gid, old_text in old_blobs.items():
        new_text = new_blobs[gid]
        if gid == "G-67":
            if not new_text.startswith(old_text):
                raise SystemExit(f"G-67 text changed before the appended cell: {gid}")
            if new_text != old_text + re.sub(r"\s+", "", G67_CELL):
                raise SystemExit("G-67 appended text is not the recorded cell")
        elif old_text != new_text:
            raise SystemExit(f"record {gid} text changed: {len(old_text)} -> {len(new_text)} chars")

    # Non-table lines must survive in order, apart from the one heading we add.
    old_prose = [l for l in lines if not l.startswith("|") and l.strip()]
    new_prose = [l for l in (pre + unsolved + post) if not l.startswith("|") and l.strip()]
    if old_prose != new_prose:
        missing = [l for l in old_prose if l not in new_prose]
        added = [l for l in new_prose if l not in old_prose]
        raise SystemExit(f"prose lines changed: missing={missing[:3]} added={added[:3]}")

    out = (
        pre
        + unsolved
        + ["", CLOSED_HEADING, "", CLOSED_HEADER, CLOSED_DELIM]
        + new_closed
        + [""]
        + post
    )
    new_text = "\n".join(out)

    print(f"  closed records: {len(new_closed)}  (G-35..G-78, ascending)")
    print(f"  padded cells (no fourth cell before): {len(padded)} -> {padded}")
    print(f"  five-cell rows collapsed to four: {len(merged)} -> {merged}")
    print(f"  fused lines split: {fused}")
    print(f"  records: {len(new_closed)}  (G-35..G-78, ascending)")
    print(f"  every record's text identical to before: measured, G-67 = prefix + recorded cell")
    print(f"  lines: {len(lines)} -> {len(out)}")
    if "--apply" not in sys.argv:
        print("  dry run; pass --apply to write")
        return 0
    LEDGER.write_text(new_text, encoding="utf-8")
    print("  written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
