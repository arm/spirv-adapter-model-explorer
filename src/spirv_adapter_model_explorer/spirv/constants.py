# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
from typing import Any

from .spirv_parser import SpirvModule
from .types import (
    ArrayType,
    Constant,
    Id,
    MatrixType,
    PointerType,
    ScalarType,
    SpirvModuleModel,
    StructType,
    TensorType,
    Type,
    VectorType,
    _operand_value,
    _set_resolved_attr,
)


def _collect_constants(module: SpirvModule) -> dict[Id, Constant]:
    constants: dict[Id, Constant] = {}
    for inst in module.instructions:
        if inst.result_id is None or inst.result_type_id is None:
            continue
        if not (
            inst.opname.startswith("OpConstant")
            or inst.opname.startswith("OpSpecConstant")
        ):
            continue
        if inst.opname in {"OpConstantTrue", "OpSpecConstantTrue"}:
            value: Any = True
        elif inst.opname in {"OpConstantFalse", "OpSpecConstantFalse"}:
            value = False
        elif inst.opname in {"OpConstantNull", "OpSpecConstantOp"}:
            value = (
                None
                if inst.opname == "OpConstantNull"
                else tuple(
                    _operand_value(operand) for operand in inst.operands[2:]
                )
            )
        elif inst.opname in {
            "OpConstantComposite",
            "OpSpecConstantComposite",
            "OpConstantCompositeReplicateEXT",
        }:
            value = tuple(
                operand.ref_id
                for operand in inst.operands[2:]
                if operand.ref_id is not None
            )
        else:
            literal_words = inst.raw_words[3:]
            value = (
                literal_words[0] if len(literal_words) == 1 else literal_words
            )
        constants[inst.result_id] = Constant(
            id=inst.result_id,
            type_id=inst.result_type_id,
            opcode=inst.opname,
            value=value,
            instruction=inst,
        )
    return constants


def _resolve_constant_values(model: SpirvModuleModel) -> None:
    resolving: set[Id] = set()
    resolved: set[Id] = set()

    def resolve(constant: Constant) -> Any:
        if constant.id in resolved:
            return constant.resolved_value
        if constant.id in resolving:
            return constant.value
        resolving.add(constant.id)
        value = constant.value
        if constant.opcode in {"OpConstant", "OpSpecConstant"}:
            decoded_value = _decode_scalar_literal_value(constant)
            if decoded_value is not _UNDECODED:
                value = decoded_value
                _set_resolved_attr(constant, "value", decoded_value)
        if constant.opcode == "OpConstantNull":
            value = _resolve_null_constant_value(constant.type, resolve)
        elif constant.opcode in {
            "OpConstantComposite",
            "OpSpecConstantComposite",
        }:
            value = tuple(
                resolve(model.constants[item])
                if isinstance(item, int) and item in model.constants
                else item
                for item in constant.value
            )
        elif constant.opcode == "OpConstantCompositeReplicateEXT":
            value = _resolve_replicated_constant_value(
                model, constant, resolve
            )
        resolving.remove(constant.id)
        resolved.add(constant.id)
        _set_resolved_attr(constant, "resolved_value", value)
        return value

    for constant in model.constants.values():
        resolve(constant)


_UNDECODED = object()


def _decode_scalar_literal_value(constant: Constant) -> object:
    type_model = constant.type
    if not isinstance(type_model, ScalarType):
        return _UNDECODED
    words = _literal_words(constant.value)
    if not words:
        return _UNDECODED
    if type_model.kind == "int":
        return _decode_integer_literal(words, type_model)
    if type_model.kind == "float":
        return _decode_float_literal(words, type_model)
    return _UNDECODED


def _literal_words(value: object) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, tuple) and all(
        isinstance(word, int) for word in value
    ):
        return value
    return ()


def _decode_integer_literal(
    words: tuple[int, ...], type_model: ScalarType
) -> object:
    width = type_model.width
    if width is None or width <= 0:
        return _UNDECODED
    word_count = (width + 31) // 32
    if len(words) < word_count:
        return _UNDECODED
    value = 0
    for index, word in enumerate(words[:word_count]):
        value |= (word & 0xFFFFFFFF) << (index * 32)
    mask = (1 << width) - 1
    value &= mask
    if type_model.signed and value & (1 << (width - 1)):
        value -= 1 << width
    return value


def _decode_float_literal(
    words: tuple[int, ...], type_model: ScalarType
) -> object:
    width = type_model.width
    if width == 16 and len(words) >= 1:
        return struct.unpack("<e", struct.pack("<H", words[0] & 0xFFFF))[0]
    if width == 32 and len(words) >= 1:
        return struct.unpack("<f", struct.pack("<I", words[0] & 0xFFFFFFFF))[0]
    if width == 64 and len(words) >= 2:
        return struct.unpack(
            "<d",
            struct.pack("<II", words[0] & 0xFFFFFFFF, words[1] & 0xFFFFFFFF),
        )[0]
    return _UNDECODED


def _resolve_null_constant_value(type_model: Type | None, resolve) -> Any:
    if isinstance(type_model, ScalarType):
        if type_model.kind == "bool":
            return False
        if type_model.kind == "int":
            return 0
        if type_model.kind == "float":
            return 0.0
        return None
    if isinstance(type_model, PointerType):
        return None
    if isinstance(type_model, VectorType):
        component = _resolve_null_constant_value(
            type_model.component_type, resolve
        )
        return tuple(component for _index in range(type_model.component_count))
    if isinstance(type_model, MatrixType):
        column = _resolve_null_constant_value(type_model.column_type, resolve)
        return tuple(column for _index in range(type_model.column_count))
    if isinstance(type_model, ArrayType):
        length = type_model.length
        length_value = resolve(length) if length is not None else None
        if isinstance(length_value, int):
            element = _resolve_null_constant_value(
                type_model.element_type, resolve
            )
            return tuple(element for _index in range(length_value))
    if isinstance(type_model, StructType):
        return tuple(
            _resolve_null_constant_value(member_type, resolve)
            for member_type in type_model.member_types
        )
    if isinstance(type_model, TensorType):
        shape = type_model.shape
        count = (
            _tensor_element_count(resolve(shape))
            if shape is not None
            else None
        )
        if count is not None:
            element = _resolve_null_constant_value(
                type_model.element_type, resolve
            )
            return tuple(element for _index in range(count))
    return None


def _resolve_replicated_constant_value(
    model: SpirvModuleModel, constant: Constant, resolve
) -> Any:
    constituent_id = next(
        (
            operand.ref_id
            for operand in constant.instruction.operands[2:]
            if operand.ref_id is not None
        ),
        None,
    )
    if constituent_id not in model.constants:
        return constant.value
    constituent_value = resolve(model.constants[constituent_id])
    type_model = model.types.get(constant.type_id)
    count = _composite_element_count(type_model, resolve)
    return (
        tuple(constituent_value for _index in range(count))
        if count
        else constituent_value
    )


def _composite_element_count(type_model: Type | None, resolve) -> int | None:
    if isinstance(type_model, VectorType):
        return type_model.component_count
    if isinstance(type_model, MatrixType):
        return type_model.column_count
    if isinstance(type_model, ArrayType):
        length = type_model.length
        if length is not None:
            length_value = resolve(length)
            if isinstance(length_value, int):
                return length_value
    if isinstance(type_model, StructType):
        return len(type_model.member_type_ids)
    if isinstance(type_model, TensorType):
        shape = type_model.shape
        if shape is not None:
            return _tensor_element_count(resolve(shape))
    return None


def _tensor_element_count(shape: object) -> int | None:
    if isinstance(shape, int):
        return shape
    if not isinstance(shape, tuple):
        return None
    count = 1
    for dimension in shape:
        if not isinstance(dimension, int):
            return None
        count *= dimension
    return count
