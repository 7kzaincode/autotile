"""
Build a parts list from a bricks payload.

Counts bricks by (kind, brick_type, color) and emits structured data suitable
for CSV download or upload to a BrickLink Wanted List.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from xml.sax.saxutils import escape

from brick_catalog import CATALOG


def _palette_by_id(palette: list[dict]) -> dict[int, dict]:
    return {p["id"]: p for p in palette}


def parts_list(payload: dict) -> list[dict]:
    palette = _palette_by_id(payload.get("palette", []))
    counts: Counter = Counter()
    for b in payload.get("bricks", []):
        kind = b.get("kind", "brick")
        counts[(kind, b["brick_type"], b["color"])] += 1

    rows: list[dict] = []
    for (kind, brick_type, color_id), qty in counts.most_common():
        cat = CATALOG.get((kind, brick_type)) or {}
        col = palette.get(color_id, {})
        rows.append({
            "kind":            kind,
            "brick_type":      brick_type,
            "ldraw_part":      cat.get("ldraw", "?"),
            "bricklink_part":  cat.get("bricklink", "?"),
            "part_name":       cat.get("name", f"{kind} {brick_type}"),
            "color_id":        color_id,
            "color_name":      col.get("name", "Unknown"),
            "ldraw_color":     col.get("ldraw"),
            "bricklink_color": col.get("bricklink"),
            "quantity":        qty,
        })
    return rows


def parts_list_csv(payload: dict) -> str:
    rows = parts_list(payload)
    buf = io.StringIO()
    fieldnames = [
        "quantity", "part_name", "kind", "brick_type", "color_name",
        "ldraw_part", "ldraw_color", "bricklink_part", "bricklink_color",
    ]
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def parts_list_bricklink_xml(payload: dict) -> str:
    rows = parts_list(payload)
    parts = ["<INVENTORY>"]
    for r in rows:
        parts.append("  <ITEM>")
        parts.append("    <ITEMTYPE>P</ITEMTYPE>")
        parts.append(f"    <ITEMID>{escape(str(r['bricklink_part']))}</ITEMID>")
        if r.get("bricklink_color") is not None:
            parts.append(f"    <COLOR>{escape(str(r['bricklink_color']))}</COLOR>")
        parts.append(f"    <MINQTY>{r['quantity']}</MINQTY>")
        parts.append(f"    <NOTIFY>N</NOTIFY>")
        parts.append("  </ITEM>")
    parts.append("</INVENTORY>")
    return "\n".join(parts)


def summary(payload: dict) -> dict:
    rows = parts_list(payload)
    return {
        "total_bricks": sum(r["quantity"] for r in rows),
        "unique_parts": len(rows),
        "by_kind": {
            k: sum(r["quantity"] for r in rows if r["kind"] == k)
            for k in sorted({r["kind"] for r in rows})
        },
        "by_brick_type": {
            bt: sum(r["quantity"] for r in rows if r["brick_type"] == bt)
            for bt in sorted({r["brick_type"] for r in rows})
        },
        "by_color": {
            r["color_name"]: r["quantity"]
            for r in sorted(rows, key=lambda x: -x["quantity"])
        },
    }
