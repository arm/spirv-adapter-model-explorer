# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

from model_explorer import graph_builder as gb

from .formatting import (
    INTEGER_INTERPRETATION_SIGNED,
    INTEGER_INTERPRETATION_TYPE,
    INTEGER_INTERPRETATION_UNSIGNED,
    IntegerInterpretation,
)
from .spirv.types import GraphArmOperation, SpirvModuleModel, StructType

TOSA_EXT_INST_SET = "TOSA.001000.1"
TOSA_EXT_INST_SET_RE = re.compile(r"^TOSA\.(\d{6})\.(\d+)$")
TOSA_MULTI_OUTPUT_TAGS: dict[str, tuple[str, ...]] = {
    "FFT2D": ("output_real", "output_imag"),
    "RFFT2D": ("output_real", "output_imag"),
}


def tosa_spec_url(instruction_set: str, opname: str) -> str | None:
    match = TOSA_EXT_INST_SET_RE.match(instruction_set)
    if match is None:
        return None

    version, patch = match.groups()
    major = int(version[:3])
    minor = int(version[3:])
    patch_number = int(patch)
    anchor = f"#_{opname.lower()}"
    return (
        "https://www.mlplatform.org/tosa/"
        f"tosa_spec_{major}_{minor}_{patch_number}.html{anchor}"
    )


def operation_tosa_spec_attr(
    operation: GraphArmOperation,
) -> gb.KeyValue | None:
    if operation.ext_inst_set_name is None:
        return None
    spec_url = tosa_spec_url(
        operation.ext_inst_set_name, operation.opcode_name
    )
    if spec_url is None:
        return None
    return gb.KeyValue(
        key="spec",
        value=f"[TOSA spec: {operation.opcode_name}]({spec_url})",
    )


def operation_output_tags(
    operation: GraphArmOperation,
) -> tuple[str | None, ...]:
    output_count = (
        len(operation.type.member_types)
        if isinstance(operation.type, StructType)
        else 1
    )
    tags = TOSA_MULTI_OUTPUT_TAGS.get(operation.opcode_name, ())
    return tuple(
        tags[index] if index < len(tags) else None
        for index in range(output_count)
    )


def constant_operand_integer_interpretation(
    operation: GraphArmOperation,
    operand_index: int,
    model: SpirvModuleModel,
) -> IntegerInterpretation:
    # Outside TOSA, keep the normal SPIR-V interpretation: signedness is taken
    # from OpTypeInt. Inside TOSA, the same OpTypeInt N 0 encoding usually means
    # TOSA signless integer, so display it as signed by default.
    if operation.ext_inst_set_name != TOSA_EXT_INST_SET:
        return INTEGER_INTERPRETATION_TYPE
    if operation.opcode_name != "RESCALE":
        return INTEGER_INTERPRETATION_SIGNED

    # RESCALE is special because it is the only TOSA 1.0.1 operator that can
    # explicitly treat signless integer values as unsigned. Its boolean flags
    # affect the data operand and the corresponding zero-point operand.
    operand_name = operation_operand_name(operation, operand_index)
    if operand_name in {"input", "input_zp"}:
        return rescale_unsigned_flag_interpretation(
            operation, model, "input_unsigned"
        )
    if operand_name in {"output", "output_zp"}:
        return rescale_unsigned_flag_interpretation(
            operation, model, "output_unsigned"
        )
    return INTEGER_INTERPRETATION_SIGNED


def rescale_unsigned_flag_interpretation(
    operation: GraphArmOperation,
    model: SpirvModuleModel,
    flag_name: str,
) -> IntegerInterpretation:
    # RESCALE's unsigned flags are compile-time boolean operands. If a malformed
    # module omits the flag or uses a non-constant value, prefer the TOSA default
    # signed/signless interpretation rather than guessing unsigned.
    flag_value = operation_constant_operand_value(operation, model, flag_name)
    return (
        INTEGER_INTERPRETATION_UNSIGNED
        if flag_value is True or flag_value == 1
        else INTEGER_INTERPRETATION_SIGNED
    )


def operation_constant_operand_value(
    operation: GraphArmOperation,
    model: SpirvModuleModel,
    operand_name: str,
) -> object:
    operand_specs = (
        operation.ext_inst_spec.operands
        if operation.ext_inst_spec is not None
        else ()
    )
    for index, spec in enumerate(operand_specs):
        if spec.name != operand_name or index >= len(operation.operand_ids):
            continue
        constant = model.constants.get(operation.operand_ids[index])
        return None if constant is None else constant.resolved_value
    return None


def operation_operand_name(
    operation: GraphArmOperation, operand_index: int
) -> str | None:
    if operation.ext_inst_spec is None:
        return None
    if operand_index >= len(operation.ext_inst_spec.operands):
        return None
    return operation.ext_inst_spec.operands[operand_index].name


def operation_input_tags(
    operation: GraphArmOperation,
) -> tuple[str | None, ...]:
    operand_specs = (
        operation.ext_inst_spec.operands
        if operation.ext_inst_spec is not None
        else ()
    )
    tags: list[str | None] = []
    operand_index = 0
    for spec_index, operand_spec in enumerate(operand_specs):
        if operand_index >= len(operation.operand_ids):
            break
        remaining_required = minimum_required_operands(
            operand_specs[spec_index + 1 :]
        )
        if operand_spec.quantifier == "*":
            repeat_count = max(
                0,
                len(operation.operand_ids)
                - operand_index
                - remaining_required,
            )
            tags.extend(
                repeated_operand_tag(operand_spec.name, repeat_index)
                for repeat_index in range(repeat_count)
            )
            operand_index += repeat_count
        elif operand_spec.quantifier == "?":
            if len(operation.operand_ids) - operand_index > remaining_required:
                tags.append(operand_spec.name)
                operand_index += 1
        else:
            tags.append(operand_spec.name)
            operand_index += 1
    tags.extend(None for _ in range(len(operation.operand_ids) - len(tags)))
    return tuple(tags)


def minimum_required_operands(operand_specs) -> int:
    return sum(
        1
        for operand_spec in operand_specs
        if operand_spec.quantifier not in {"?", "*"}
    )


def repeated_operand_tag(name: str | None, repeat_index: int) -> str | None:
    if not name:
        return None
    match = re.fullmatch(r"(.*?)(\d+)", name)
    if match is None:
        return f"{name}{repeat_index}"
    prefix, suffix = match.groups()
    return f"{prefix}{int(suffix) + repeat_index}"
