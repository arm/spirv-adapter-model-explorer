# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from model_explorer import graph_builder as gb

from .assembly import SpirvDisassemblyLine
from .disassembly_metadata import (
    disassembly_attrs,
    graph_group_attrs,
)
from .spirv.spirv_parser import SpirvInstruction, SpirvModule


def build_raw_instruction_graph(
    module: SpirvModule,
    *,
    disassembly_text: str | None = None,
    disassembly_lines_by_byte_offset: dict[int, SpirvDisassemblyLine]
    | None = None,
) -> gb.Graph:
    nodes = [
        raw_instruction_node(
            inst,
            disassembly_lines_by_byte_offset=disassembly_lines_by_byte_offset,
        )
        for inst in module.instructions
    ]
    return gb.Graph(
        id="module",
        nodes=nodes,
        groupNodeAttributes=graph_group_attrs({}, disassembly_text),
    )


def raw_instruction_node(
    inst: SpirvInstruction,
    *,
    disassembly_lines_by_byte_offset: dict[int, SpirvDisassemblyLine]
    | None = None,
) -> gb.GraphNode:
    return gb.GraphNode(
        id=f"inst_{inst.index}",
        label=inst.opname,
        attrs=[
            gb.KeyValue(key="opcode", value=inst.opname),
            gb.KeyValue(
                key="operands",
                value=", ".join(operand.display for operand in inst.operands),
            ),
            gb.KeyValue(key="spv word offset", value=str(inst.word_offset)),
            gb.KeyValue(key="spv word count", value=str(inst.word_count)),
            gb.KeyValue(key="spv opcode", value=str(inst.opcode)),
            *disassembly_attrs(inst, disassembly_lines_by_byte_offset),
        ],
    )
