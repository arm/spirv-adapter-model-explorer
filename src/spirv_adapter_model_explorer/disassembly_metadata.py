# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from model_explorer import graph_builder as gb

from .assembly import SpirvDisassemblyLine
from .spirv.spirv_parser import SpirvInstruction


def graph_group_attrs(
    attrs: dict[str, str], disassembly_text: str | None
) -> dict[str, dict[str, str]]:
    group_attrs = dict(attrs)
    if disassembly_text is not None:
        group_attrs["spv full asm"] = disassembly_text
    return {"": group_attrs}


def disassembly_attrs(
    inst: SpirvInstruction,
    lines_by_byte_offset: dict[int, SpirvDisassemblyLine] | None,
) -> list[gb.KeyValue]:
    line = disassembly_line(inst, lines_by_byte_offset)
    if line is None:
        return []
    return [
        gb.KeyValue(key="spv asm line", value=str(line.line_number)),
        gb.KeyValue(key="spv byte offset", value=f"0x{line.byte_offset:08x}"),
        gb.KeyValue(key="spv asm", value=line.text),
    ]


def disassembly_line(
    inst: SpirvInstruction,
    lines_by_byte_offset: dict[int, SpirvDisassemblyLine] | None,
) -> SpirvDisassemblyLine | None:
    if lines_by_byte_offset is None:
        return None
    return lines_by_byte_offset.get(inst.word_offset * 4)
