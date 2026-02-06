#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a 7-page LinkedIn storytelling carousel as a PDF (no external deps)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap


@dataclass
class Slide:
    title: str
    subtitle: str
    bullets: list[str]
    kicker: str


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _line_cmd(x: int, y: int, text: str, font: str = "F1", size: int = 30, rgb: tuple[float, float, float] = (0, 0, 0)) -> str:
    r, g, b = rgb
    return f"BT /{font} {size} Tf {r:.3f} {g:.3f} {b:.3f} rg 1 0 0 1 {x} {y} Tm ({_esc(text)}) Tj ET\n"


def _rect_cmd(x: int, y: int, w: int, h: int, rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    return f"{r:.3f} {g:.3f} {b:.3f} rg {x} {y} {w} {h} re f\n"


def _slide_stream(slide: Slide, idx: int, total: int, width: int, height: int) -> bytes:
    top_h = 210
    left = 80
    y = height - 115

    out = []
    out.append(_rect_cmd(0, 0, width, height, (0.965, 0.973, 0.992)))
    out.append(_rect_cmd(0, height - top_h, width, top_h, (0.050, 0.220, 0.420)))
    out.append(_rect_cmd(0, height - top_h, 16, top_h, (0.980, 0.250, 0.250)))

    out.append(_line_cmd(left, y, slide.title, font="F2", size=52, rgb=(1, 1, 1)))
    out.append(_line_cmd(left, y - 62, slide.subtitle, font="F1", size=24, rgb=(0.840, 0.910, 1.000)))

    y = height - top_h - 72
    if slide.kicker:
        out.append(_line_cmd(left, y, slide.kicker, font="F2", size=26, rgb=(0.050, 0.220, 0.420)))
        y -= 58

    for bullet in slide.bullets:
        wrapped = textwrap.wrap(bullet, width=62) or [bullet]
        out.append(_line_cmd(left, y, f"- {wrapped[0]}", font="F1", size=28, rgb=(0.110, 0.160, 0.220)))
        y -= 40
        for cont in wrapped[1:]:
            out.append(_line_cmd(left + 26, y, cont, font="F1", size=28, rgb=(0.110, 0.160, 0.220)))
            y -= 38
        y -= 14

    out.append(_line_cmd(left, 54, f"Slide {idx}/{total}", font="F1", size=18, rgb=(0.35, 0.42, 0.54)))
    out.append(_line_cmd(width - 440, 54, "Portfolio Analytics | Trade Republic", font="F1", size=18, rgb=(0.35, 0.42, 0.54)))
    return "".join(out).encode("latin-1", errors="replace")


def _build_pdf(slides: list[Slide], out_path: Path) -> None:
    width, height = 1080, 1350
    obj_map: dict[int, bytes] = {}

    # Static font objects.
    obj_map[1] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    obj_map[2] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"

    next_id = 5
    page_ids: list[int] = []

    for idx, slide in enumerate(slides, start=1):
        content = _slide_stream(slide, idx, len(slides), width, height)
        content_id = next_id
        page_id = next_id + 1
        next_id += 2

        stream = (
            b"<< /Length " + str(len(content)).encode("ascii") + b" >>\n"
            b"stream\n" + content + b"endstream"
        )
        obj_map[content_id] = stream

        page_obj = (
            f"<< /Type /Page /Parent 3 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 1 0 R /F2 2 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("ascii")
        obj_map[page_id] = page_obj
        page_ids.append(page_id)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    obj_map[3] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [ {kids} ] >>".encode("ascii")
    obj_map[4] = b"<< /Type /Catalog /Pages 3 0 R >>"

    ids = sorted(obj_map.keys())
    parts = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets: dict[int, int] = {}

    for oid in ids:
        offsets[oid] = sum(len(p) for p in parts)
        parts.append(f"{oid} 0 obj\n".encode("ascii"))
        parts.append(obj_map[oid] + b"\n")
        parts.append(b"endobj\n")

    xref_pos = sum(len(p) for p in parts)
    max_id = max(ids)
    parts.append(f"xref\n0 {max_id + 1}\n".encode("ascii"))
    parts.append(b"0000000000 65535 f \n")
    for oid in range(1, max_id + 1):
        off = offsets.get(oid, 0)
        parts.append(f"{off:010d} 00000 n \n".encode("ascii"))

    parts.append(
        (
            "trailer\n"
            f"<< /Size {max_id + 1} /Root 4 0 R >>\n"
            "startxref\n"
            f"{xref_pos}\n"
            "%%EOF\n"
        ).encode("ascii")
    )

    out_path.write_bytes(b"".join(parts))


def main() -> None:
    slides = [
        Slide(
            title="Trade Republic, but deeper",
            subtitle="From simple statement to portfolio intelligence",
            kicker="Hook",
            bullets=[
                "Investing is easy. Understanding true performance and risk is much harder.",
                "Portfolio Analytics was built to close that gap.",
            ],
        ),
        Slide(
            title="The problem",
            subtitle="Retail reporting is often fragmented",
            kicker="Why this matters",
            bullets=[
                "You see movements, but not real contributors to PnL.",
                "Risk is visible only at surface level.",
                "It is hard to share a clean, decision-ready view.",
            ],
        ),
        Slide(
            title="The solution",
            subtitle="One statement in, full dashboard out",
            kicker="Workflow",
            bullets=[
                "Upload one Trade Republic Account Statement PDF.",
                "Get overview, performance, risk, advanced risk and diagnostics.",
                "A professional PDF report is generated automatically.",
            ],
        ),
        Slide(
            title="Trust by design",
            subtitle="Stateless and session-only processing",
            kicker="Privacy first",
            bullets=[
                "No persistent storage of uploaded statements.",
                "No persistent storage of generated reports.",
                "Clear transparency on external data enrichment and limits.",
            ],
        ),
        Slide(
            title="What you can read immediately",
            subtitle="Not just charts: decision-grade analytics",
            kicker="Analytical depth",
            bullets=[
                "TWR, drawdown, rolling returns and contributors.",
                "ETF and equity exposures with diagnostics.",
                "Tail risk, VaR, rolling correlations and volatility views.",
            ],
        ),
        Slide(
            title="No black box",
            subtitle="Assumptions are explicit",
            kicker="Methodology transparency",
            bullets=[
                "Italian baseline tax model on capital gains (26%).",
                "Carry-forward loss logic where applicable.",
                "Fallback rules for missing prices or delisted instruments.",
            ],
        ),
        Slide(
            title="Mission",
            subtitle="Asset-manager discipline for retail portfolios",
            kicker="Call to action",
            bullets=[
                "Less noise. Better decisions.",
                "If you want to test it or review the methodology, reach out.",
                "Next step: richer benchmarks and scenario analysis.",
            ],
        ),
    ]

    out_path = Path("linkedin_carousel.pdf")
    _build_pdf(slides, out_path)
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    main()
