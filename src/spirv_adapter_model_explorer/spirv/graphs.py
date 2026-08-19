# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .spirv_grammar import SpirvGrammar
from .spirv_parser import SpirvInstruction, SpirvModule
from .types import (
    DataGraph,
    DataGraphEntryPoint,
    DataGraphInput,
    DataGraphOperation,
    DataGraphOutput,
    ExtInstImport,
    Id,
    _literal_int,
    _literal_string,
)


def _collect_data_graph_entry_points(
    module: SpirvModule,
) -> tuple[DataGraphEntryPoint, ...]:
    entry_points: list[DataGraphEntryPoint] = []
    for inst in module.instructions:
        if inst.opname != "OpGraphEntryPointARM" or len(inst.operands) < 2:
            continue
        graph_id = inst.operands[0].ref_id
        if graph_id is None:
            continue
        entry_points.append(
            DataGraphEntryPoint(
                graph_id=graph_id,
                name=_literal_string(inst.operands[1].display),
                interface_ids=tuple(
                    operand.ref_id
                    for operand in inst.operands[2:]
                    if operand.ref_id is not None
                ),
                instruction=inst,
            )
        )
    return tuple(entry_points)


def _collect_data_graphs(
    module: SpirvModule,
    grammar: SpirvGrammar,
    extended_instruction_imports: dict[Id, ExtInstImport],
) -> dict[Id, DataGraph]:
    data_graphs: dict[Id, DataGraph] = {}
    current_data_graph: SpirvInstruction | None = None
    body_instructions: list[SpirvInstruction] = []
    inputs: list[DataGraphInput] = []
    operations: list[DataGraphOperation] = []
    outputs: list[DataGraphOutput] = []

    def finish_data_graph() -> None:
        nonlocal \
            current_data_graph, \
            body_instructions, \
            inputs, \
            operations, \
            outputs
        if (
            current_data_graph is not None
            and current_data_graph.result_id is not None
            and current_data_graph.result_type_id is not None
        ):
            data_graphs[current_data_graph.result_id] = DataGraph(
                id=current_data_graph.result_id,
                type_id=current_data_graph.result_type_id,
                instruction=current_data_graph,
                body_instructions=tuple(body_instructions),
                inputs=tuple(inputs),
                operations=tuple(operations),
                outputs=tuple(outputs),
            )
        current_data_graph = None
        body_instructions = []
        inputs = []
        operations = []
        outputs = []

    for inst in module.instructions:
        if inst.opname == "OpGraphARM":
            finish_data_graph()
            current_data_graph = inst
            continue
        if current_data_graph is None:
            continue
        if inst.opname == "OpGraphEndARM":
            body_instructions.append(inst)
            finish_data_graph()
            continue

        body_instructions.append(inst)
        if (
            inst.opname == "OpGraphInputARM"
            and inst.result_id is not None
            and inst.result_type_id is not None
            and len(inst.operands) >= 3
            and inst.operands[2].ref_id is not None
        ):
            element_index_id = (
                inst.operands[3].ref_id if len(inst.operands) >= 4 else None
            )
            inputs.append(
                DataGraphInput(
                    id=inst.result_id,
                    type_id=inst.result_type_id,
                    input_index_id=inst.operands[2].ref_id,
                    element_index_id=element_index_id,
                    instruction=inst,
                )
            )
        elif inst.opname == "OpGraphSetOutputARM" and len(inst.operands) >= 2:
            value_id = inst.operands[0].ref_id
            output_index_id = inst.operands[1].ref_id
            element_index_id = (
                inst.operands[2].ref_id if len(inst.operands) >= 3 else None
            )
            if value_id is not None and output_index_id is not None:
                outputs.append(
                    DataGraphOutput(
                        value_id=value_id,
                        output_index_id=output_index_id,
                        element_index_id=element_index_id,
                        instruction=inst,
                    )
                )
        elif (
            inst.result_id is not None
            and inst.result_type_id is not None
            and not _is_data_graph_debug_instruction(
                inst, extended_instruction_imports
            )
        ):
            operations.append(
                _data_graph_operation_from_instruction(
                    inst, grammar, extended_instruction_imports
                )
            )

    finish_data_graph()
    return data_graphs


def _is_data_graph_debug_instruction(
    inst: SpirvInstruction,
    extended_instruction_imports: dict[Id, ExtInstImport],
) -> bool:
    if inst.opname != "OpExtInst" or len(inst.operands) < 4:
        return False
    ext_inst_set_id = inst.operands[2].ref_id
    if ext_inst_set_id is None:
        return False
    ext_inst_set = extended_instruction_imports.get(ext_inst_set_id)
    return ext_inst_set is not None and ext_inst_set.name.startswith(
        "NonSemantic.Graph.DebugInfo."
    )


def _data_graph_operation_from_instruction(
    inst: SpirvInstruction,
    grammar: SpirvGrammar,
    extended_instruction_imports: dict[Id, ExtInstImport],
) -> DataGraphOperation:
    assert inst.result_id is not None
    assert inst.result_type_id is not None

    if inst.opname == "OpExtInst" and len(inst.operands) >= 4:
        ext_inst_set_id = inst.operands[2].ref_id
        ext_inst_opcode = _literal_int(inst.operands[3].display)
        ext_inst_set_name = (
            extended_instruction_imports[ext_inst_set_id].name
            if ext_inst_set_id in extended_instruction_imports
            else None
        )
        ext_inst_spec = (
            grammar.extinst_instruction_spec(
                ext_inst_set_name, ext_inst_opcode
            )
            if ext_inst_set_name is not None and ext_inst_opcode is not None
            else None
        )
        operand_ids = tuple(
            operand.ref_id
            for operand in inst.operands[4:]
            if operand.ref_id is not None
        )
        opcode_name = (
            ext_inst_spec.opname
            if ext_inst_spec is not None
            else f"OpExtInst {ext_inst_opcode}"
            if ext_inst_opcode is not None
            else "OpExtInst"
        )
        return DataGraphOperation(
            id=inst.result_id,
            type_id=inst.result_type_id,
            opcode_name=opcode_name,
            operand_ids=operand_ids,
            instruction=inst,
            ext_inst_set_id=ext_inst_set_id,
            ext_inst_opcode=ext_inst_opcode,
            ext_inst_set_name=ext_inst_set_name,
            ext_inst_spec=ext_inst_spec,
        )

    return DataGraphOperation(
        id=inst.result_id,
        type_id=inst.result_type_id,
        opcode_name=inst.opname,
        operand_ids=tuple(
            operand.ref_id
            for operand in inst.operands
            if operand.ref_id is not None
        ),
        instruction=inst,
    )
