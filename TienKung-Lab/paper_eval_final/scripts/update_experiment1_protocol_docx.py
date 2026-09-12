#!/usr/bin/env python3
"""Update Experiment 1 stratification wording in the protocol DOCX."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOCX = Path(__file__).resolve().parents[3] / "人形机器人多步可恢复性算法实验评测协议.docx"


def text(paragraph: ET.Element) -> str:
    return "".join(node.text or "" for node in paragraph.iter(f"{W}t"))


def replace(paragraph: ET.Element, value: str) -> None:
    nodes = list(paragraph.iter(f"{W}t"))
    nodes[0].text = value
    for node in nodes[1:]:
        node.text = ""


def main() -> None:
    exact = {
        "先收集 Nmin=2、3、4 各 120 条证书有效样本并合并计算共享 q1、q2；再按共享边界将九个 Nmin-margin 单元补齐至各 40 条":
            "固定Nmin=2、3、4各120条证书有效校准样本，分别计算q_n1、q_n2；九个Nmin-margin相对等级单元各40条",
        "Low": "Lower", "margin < q1": "margin < q_n1",
        "Medium": "Middle", "q1 ≤ margin < q2": "q_n1 ≤ margin < q_n2",
        "High": "Upper", "margin ≥ q2": "margin ≥ q_n2",
        "共享 pilot 边界": "各Nmin独立pilot边界",
        "实验一以共享边界下九单元等样本分析集为主；实验二和实验三仍以固定预算结果为主":
            "实验一以各Nmin独立三分位下九单元等样本分析集为主；实验二和实验三仍以固定预算结果为主",
        "图 A 横轴为共享 q1、q2 定义的 Low、Medium、High margin，纵轴为持续恢复率，Nmin = 2、3、4 为三条折线，并给出 Wilson 95% CI":
            "图A横轴为各Nmin内部的Lower、Middle、Upper margin相对三分位，纵轴为持续恢复率，Nmin=2、3、4为三条折线，给出Wilson 95% CI并列出各Nmin实际边界",
        "TD0 time certificate valid Nmin margin raw margin group shared q1 shared q2 sample role calibration only margin degenerate calibration manifest hash evaluation manifest hash":
            "TD0 time certificate valid Nmin margin raw margin group Nmin q1 Nmin q2 sample role calibration only margin degenerate calibration manifest hash evaluation manifest hash",
        "理论实验使用同一套共享 q1、q2 在每个 Nmin 内比较三档 margin，避免改变等级含义或把 Nmin 的影响误归因于 margin":
            "理论实验在每个Nmin内使用独立冻结的q_n1、q_n2比较三档相对margin；不同Nmin的等级不表示相同绝对margin范围",
    }
    prefix = "使用由平衡pilot校准数据确定的共享三分位边界划分margin。"
    long_value = (
        "使用原冻结校准manifest中Nmin=2、3、4各120条证书有效试验，在每个Nmin内独立计算margin三分位边界。"
        "对每组120个原始margin排序，q_n1=(m_n(40)+m_n(41))/2，q_n2=(m_n(80)+m_n(81))/2。"
        "Lower为margin<q_n1，Middle为q_n1≤margin<q_n2，Upper为margin≥q_n2。"
        "这些等级仅表示各Nmin支持域内的相对三分位，不代表相同绝对margin范围。六个边界在正式实验前冻结且不得重新计算。"
        "校准trial不得根据恢复结果重选；若边界处原始margin相同或q_n1≥q_n2，则标记MARGIN_DEGENERATE并停止。"
        "Pilot九个单元各40条，并尽量满足每坡度8条、每速度组20条、每冲量组20条、每方向10条；选择和补样不得使用恢复、生存或失败结果。"
    )
    with ZipFile(DOCX, "r") as source:
        root = ET.fromstring(source.read("word/document.xml"))
        counts, long_count = {key: 0 for key in exact}, 0
        for paragraph in root.iter(f"{W}p"):
            value = text(paragraph)
            if value.startswith(prefix):
                replace(paragraph, long_value); long_count += 1
            elif value in exact:
                replace(paragraph, exact[value]); counts[value] += 1
        all_text = [text(paragraph) for paragraph in root.iter(f"{W}p")]
        missing = [old for old, new in exact.items() if counts[old] != 1 and new not in all_text]
        if (long_count != 1 and long_value not in all_text) or missing:
            raise ValueError(f"unexpected DOCX match counts: long={long_count}, exact={counts}")
        document = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{DOCX.name}.", dir=DOCX.parent); os.close(fd)
        try:
            with ZipFile(temporary, "w", ZIP_DEFLATED) as target:
                for info in source.infolist():
                    target.writestr(info, document if info.filename == "word/document.xml" else source.read(info.filename))
            os.replace(temporary, DOCX)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)


if __name__ == "__main__":
    main()
