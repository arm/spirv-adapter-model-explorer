# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .spirv_parser import SpirvInstruction, SpirvModule
from .types import (
    BasicBlock,
    Function,
    FunctionParameter,
    Id,
    _enum_display_name,
)


def _collect_functions(module: SpirvModule) -> dict[Id, Function]:
    functions: dict[Id, Function] = {}
    current_function: SpirvInstruction | None = None
    parameters: list[FunctionParameter] = []
    blocks: list[BasicBlock] = []
    current_label: SpirvInstruction | None = None
    current_block_instructions: list[SpirvInstruction] = []

    def finish_block() -> None:
        nonlocal current_label, current_block_instructions
        if current_label is None or current_label.result_id is None:
            return
        terminator = _find_terminator(current_block_instructions)
        blocks.append(
            BasicBlock(
                id=current_label.result_id,
                label=current_label,
                instructions=tuple(current_block_instructions),
                terminator=terminator,
                successors=_successors(terminator),
            )
        )
        current_label = None
        current_block_instructions = []

    def finish_function() -> None:
        nonlocal current_function, parameters, blocks
        finish_block()
        if (
            current_function is not None
            and current_function.result_id is not None
            and current_function.result_type_id is not None
            and len(current_function.operands) >= 4
        ):
            function_type_id = current_function.operands[3].ref_id
            if function_type_id is not None:
                functions[current_function.result_id] = Function(
                    id=current_function.result_id,
                    return_type_id=current_function.result_type_id,
                    function_type_id=function_type_id,
                    control=_enum_display_name(
                        current_function.operands[2].display
                    ),
                    parameters=tuple(parameters),
                    blocks=tuple(blocks),
                    instruction=current_function,
                )
        current_function = None
        parameters = []
        blocks = []

    for inst in module.instructions:
        if inst.opname == "OpFunction":
            finish_function()
            current_function = inst
        elif inst.opname == "OpFunctionParameter" and current_function:
            if inst.result_id is not None and inst.result_type_id is not None:
                parameters.append(
                    FunctionParameter(
                        inst.result_id, inst.result_type_id, inst
                    )
                )
        elif inst.opname == "OpLabel" and current_function:
            finish_block()
            current_label = inst
        elif inst.opname == "OpFunctionEnd" and current_function:
            finish_function()
        elif current_label is not None:
            current_block_instructions.append(inst)

    finish_function()
    return functions


def _find_terminator(
    instructions: list[SpirvInstruction],
) -> SpirvInstruction | None:
    return next(
        (inst for inst in reversed(instructions) if _is_terminator(inst)), None
    )


def _is_terminator(inst: SpirvInstruction) -> bool:
    return inst.opname in {
        "OpBranch",
        "OpBranchConditional",
        "OpSwitch",
        "OpKill",
        "OpReturn",
        "OpReturnValue",
        "OpUnreachable",
        "OpTerminateInvocation",
    }


def _successors(terminator: SpirvInstruction | None) -> tuple[Id, ...]:
    if terminator is None:
        return ()
    if terminator.opname == "OpBranch":
        return tuple(
            operand.ref_id
            for operand in terminator.operands[:1]
            if operand.ref_id is not None
        )
    if terminator.opname == "OpBranchConditional":
        return tuple(
            operand.ref_id
            for operand in terminator.operands[1:3]
            if operand.ref_id is not None
        )
    if terminator.opname == "OpSwitch":
        return tuple(
            operand.ref_id
            for operand in terminator.operands[1:]
            if operand.ref_id is not None
        )
    return ()
