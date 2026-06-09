"""
从更新后的实验报告 Markdown 生成 docx。

用法：
  python ignore/tools/build_report_docx.py

依赖：
  pip install python-docx
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

# ── 路径 ──
ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "3_Document"
MD = DOCS / "基于透视校正的平面目标图形识别与尺寸测量方法研究.md"
# 使用已排版完成的 docx 作为格式模板
TEMPLATE = DOCS / "2315102026 宋嘉诚 基于透视校正的平面目标图形识别与尺寸测量方法研究-排版完成.docx"
OUT = DOCS / "_rendered_paper" / "2315102026 宋嘉诚 基于PnP的单目视觉距离测量方法.docx"


# ── 字体工具 ──

def set_run_font(run, east_asia="宋体", ascii_font="Times New Roman", size=None, bold=None):
    run.font.name = ascii_font
    rfonts = run._element.get_or_add_rPr().get_or_add_rFonts()
    rfonts.set(qn("w:ascii"), ascii_font)
    rfonts.set(qn("w:hAnsi"), ascii_font)
    rfonts.set(qn("w:eastAsia"), east_asia)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def set_paragraph_fonts(paragraph, east_asia="宋体", ascii_font="Times New Roman", size=10.5):
    for run in paragraph.runs:
        set_run_font(run, east_asia=east_asia, ascii_font=ascii_font, size=size)


# ── 文档操作 ──

def clear_document_body(doc):
    body = doc._body._element
    sect_pr = None
    if len(body) and body[-1].tag == qn("w:sectPr"):
        sect_pr = copy.deepcopy(body[-1])
    for child in list(body):
        body.remove(child)
    if sect_pr is not None:
        body.append(sect_pr)


def configure_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.size = Pt(10.5)
    normal.paragraph_format.first_line_indent = Pt(21)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)

    for name, size in [("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 10.5)]:
        st = styles[name]
        st.font.bold = True
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.first_line_indent = Pt(0)
        st.paragraph_format.space_before = Pt(6)
        st.paragraph_format.space_after = Pt(3)
        st.paragraph_format.line_spacing = 1.5

    if "Caption" in styles:
        cap = styles["Caption"]
        cap.font.size = Pt(9)
        cap.font.italic = False
        cap.font.color.rgb = RGBColor(0, 0, 0)
        cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.first_line_indent = Pt(0)


# ── Markdown 解析 ──

def parse_markdown(text: str):
    lines = text.replace("\r\n", "\n").split("\n")
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            blocks.append(("code", "\n".join(buf)))
            continue
        if line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i])
                i += 1
            blocks.append(("table", rows))
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            blocks.append(("heading", (len(m.group(1)), m.group(2).strip())))
            i += 1
            continue
        m = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
        if m:
            items = []
            while i < len(lines):
                mm = re.match(r"^\s*(\d+)\.\s+(.*)$", lines[i])
                if not mm:
                    break
                items.append(mm.group(2).strip())
                i += 1
            blocks.append(("numbered", items))
            continue
        para = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,6})\s+", lines[i]) and not lines[i].lstrip().startswith("|") and not lines[i].startswith("```") and not re.match(r"^\s*\d+\.\s+", lines[i]):
            para.append(lines[i].strip())
            i += 1
        blocks.append(("para", " ".join(para)))
    return blocks


# ── 段落 / 文本 ──

def add_mixed_text(paragraph, text, bold_prefix=False):
    if bold_prefix and "：" in text:
        prefix, rest = text.split("：", 1)
        r = paragraph.add_run(prefix + "：")
        set_run_font(r, bold=True)
        r = paragraph.add_run(rest)
        set_run_font(r)
    else:
        r = paragraph.add_run(text)
        set_run_font(r)


def add_body_para(doc, text, first_line=True, align=None):
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.first_line_indent = Pt(21) if first_line else Pt(0)
    p.paragraph_format.line_spacing = 1.5
    p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    add_mixed_text(p, text, bold_prefix=("关键词" in text[:5] or "摘" in text[:5] and "要" in text[:5]))
    return p


# ── 表格 ──

def parse_table(rows):
    parsed = []
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        parsed.append(cells)
    if len(parsed) > 1 and all(re.fullmatch(r":?-{3,}:?", c or "---") for c in parsed[1]):
        parsed.pop(1)
    width = max(len(r) for r in parsed)
    return [r + [""] * (width - len(r)) for r in parsed]


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_table_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "808080")


def add_table(doc, rows):
    data = parse_table(rows)
    table = doc.add_table(rows=len(data), cols=len(data[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_table_borders(table)
    for r_idx, row in enumerate(data):
        for c_idx, value in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if r_idx == 0:
                set_cell_shading(cell, "EDEDED")
            p = cell.paragraphs[0]
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.line_spacing = 1.2
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(value) <= 16 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(value)
            set_run_font(run, size=9, bold=(r_idx == 0))
    doc.add_paragraph()


def add_code_block(doc, text):
    for line in text.splitlines() or [""]:
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.left_indent = Pt(18)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(line)
        set_run_font(r, east_asia="宋体", ascii_font="Consolas", size=9)


# ── 封面 ──

def add_cover(doc, title):
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("课程论文")
    set_run_font(r, east_asia="黑体", ascii_font="Times New Roman", size=18, bold=True)

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    set_run_font(r, east_asia="黑体", ascii_font="Times New Roman", size=18, bold=True)

    for _ in range(3):
        doc.add_paragraph()
    info = [
        "学    院：信息科学与工程学院",
        "专    业：通信工程",
        "班    级：通信工程2023级1班",
        "姓    名：宋嘉诚",
        "学    号：2315102026",
        "任课教师：陈燕",
    ]
    for item in info:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(4.0)
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.line_spacing = 1.5
        r = p.add_run(item)
        set_run_font(r, size=12)

    for _ in range(2):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("2026年6月")
    set_run_font(r, size=12)

    doc.add_section(WD_SECTION.NEW_PAGE)


# ── 构建 ──

def build():
    text = MD.read_text(encoding="utf-8-sig")
    blocks = parse_markdown(text)
    title = blocks[0][1][1] if blocks and blocks[0][0] == "heading" else MD.stem

    doc = Document(str(TEMPLATE))
    clear_document_body(doc)
    configure_styles(doc)

    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    add_cover(doc, title)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    set_run_font(r, east_asia="黑体", ascii_font="Times New Roman", size=15, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("宋嘉诚")
    set_run_font(r, size=10.5)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("（华侨大学信息科学与工程学院，福建 厦门 362021）")
    set_run_font(r, size=10.5)

    skip_first_title = True
    in_abstract = False
    for kind, value in blocks:
        if skip_first_title and kind == "heading" and value[0] == 1:
            skip_first_title = False
            continue
        if kind == "heading":
            level, heading = value
            if heading == "摘要":
                in_abstract = True
                continue
            in_abstract = False
            if level == 1:
                style = "Heading 1"
            elif level == 2:
                style = "Heading 1" if re.match(r"^\d+\s+", heading) else "Heading 2"
            else:
                style = "Heading 3"
            p = doc.add_paragraph(style=style)
            p.paragraph_format.first_line_indent = Pt(0)
            r = p.add_run(heading)
            set_run_font(r, east_asia="黑体", ascii_font="Times New Roman", bold=True)
        elif kind == "para":
            text_value = value
            if in_abstract:
                add_body_para(doc, "摘  要：" + text_value, first_line=False)
                in_abstract = False
            elif text_value.startswith("**关键词：**"):
                text_value = text_value.replace("**关键词：**", "关键词：", 1)
                add_body_para(doc, text_value, first_line=False)
            elif text_value.startswith("**") and text_value.endswith("**"):
                p = add_body_para(doc, text_value.strip("*"), first_line=False, align=WD_ALIGN_PARAGRAPH.CENTER)
                for run in p.runs:
                    run.bold = True
            else:
                add_body_para(doc, text_value)
        elif kind == "numbered":
            for idx, item in enumerate(value, start=1):
                p = doc.add_paragraph(style="Normal")
                p.paragraph_format.first_line_indent = Pt(0)
                p.paragraph_format.left_indent = Pt(21)
                p.paragraph_format.line_spacing = 1.5
                r = p.add_run(f"{idx}. {item}")
                set_run_font(r)
        elif kind == "table":
            add_table(doc, value)
        elif kind == "code":
            add_code_block(doc, value)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print(f"✅ 已生成: {OUT}")


if __name__ == "__main__":
    build()
