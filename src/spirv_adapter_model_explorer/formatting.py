# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal, TypeGuard

from model_explorer import graph_builder as gb

from .spirv.types import (
    ArrayType,
    Constant,
    Definition,
    GraphArmType,
    MatrixType,
    RuntimeArrayType,
    ScalarType,
    SpirvModuleModel,
    StructType,
    TensorArmType,
    Type,
    VectorType,
)

IntegerInterpretation = Literal["type", "signed", "unsigned"]

# Integer literal display normally follows the SPIR-V OpTypeInt signedness bit
# ("type"). That is correct for raw/non-TOSA SPIR-V, where OpTypeInt N 0 is
# conventionally displayed as uintN.
#
# TOSA is different: the TOSA spec defines i8/i16/i32/i48 as signless integer
# values that are interpreted as signed two's-complement unless an operator says
# otherwise. The SPIR-V TOSA extended-instruction spec lowers those signless
# TOSA integer types to OpTypeInt N 0, which is indistinguishable from an
# unsigned SPIR-V type without knowing the consuming TOSA operator. Therefore
# TOSA operand attributes default to signed display ("signed") even when the
# underlying SPIR-V type has signedness 0.
#
# RESCALE is the one TOSA 1.0.1 exception for unsigned integer values. It carries
# explicit input_unsigned/output_unsigned boolean attributes; when they are true,
# the corresponding input/output values and zero points are displayed as unsigned
# ("unsigned").
INTEGER_INTERPRETATION_TYPE: IntegerInterpretation = "type"
INTEGER_INTERPRETATION_SIGNED: IntegerInterpretation = "signed"
INTEGER_INTERPRETATION_UNSIGNED: IntegerInterpretation = "unsigned"
CONSTANT_SEQUENCE_PREVIEW_LENGTH = 32


def definition_type(
    definition: Definition | None, model: SpirvModuleModel
) -> Type | None:
    if definition is None or definition.type_id is None:
        return None
    return model.types.get(definition.type_id)


def tensor_shape_metadata_value(type_model: Type | None) -> str | None:
    if not isinstance(type_model, TensorArmType):
        return None
    return format_tensor_type(type_model)


def format_tensor_type(type_model: TensorArmType) -> str:
    dimensions = tensor_shape_dimensions(type_model)
    element_type = format_element_type(type_model.element_type)
    return f"{element_type}[{', '.join(dimensions)}]"


def tensor_shape_dimensions(type_model: TensorArmType) -> tuple[str, ...]:
    if type_model.shape is None:
        rank = format_constant_value(type_model.rank)
        return (
            tuple("*" for _ in range(int(rank))) if rank.isdigit() else ("*",)
        )
    value = type_model.shape.resolved_value
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    return (str(value),)


def format_tensor_shape(type_model: TensorArmType) -> str:
    return "[" + ", ".join(tensor_shape_dimensions(type_model)) + "]"


def format_element_type(
    type_model: Type | None,
    *,
    integer_interpretation: IntegerInterpretation = INTEGER_INTERPRETATION_TYPE,
) -> str:
    if isinstance(type_model, ScalarType):
        width = "" if type_model.width is None else str(type_model.width)
        if type_model.kind == "bool":
            return "bool"
        if type_model.kind == "float":
            return f"float{width}"
        if type_model.kind == "int":
            prefix = format_integer_type_prefix(
                type_model, integer_interpretation
            )
            return f"{prefix}{width}"
        return type_model.kind
    return format_type(type_model, None)


def format_integer_type_prefix(
    type_model: ScalarType,
    integer_interpretation: IntegerInterpretation,
) -> str:
    if integer_interpretation == INTEGER_INTERPRETATION_UNSIGNED:
        return "uint"
    if integer_interpretation == INTEGER_INTERPRETATION_SIGNED:
        return "int"
    return "uint" if type_model.signed is False else "int"


def tensor_attrs(type_model: Type | None) -> list[gb.KeyValue]:
    if not isinstance(type_model, TensorArmType):
        return []
    return [
        gb.KeyValue(
            key="dtype",
            value=format_element_type(type_model.element_type),
        ),
        gb.KeyValue(
            key="rank",
            value=format_constant_value(type_model.rank),
        ),
        gb.KeyValue(
            key="shape",
            value=format_tensor_shape(type_model),
        ),
    ]


def format_id(id: int | None) -> str:
    return "" if id is None else f"%{id}"


def definition_name_attrs(
    key: str,
    id: int,
    definition: Definition | None,
    model: SpirvModuleModel,
) -> list[gb.KeyValue]:
    name = definition_name(id, definition, model)
    if name is None:
        return []
    return [gb.KeyValue(key=key, value=name)]


def definition_name(
    id: int, definition: Definition | None, model: SpirvModuleModel
) -> str | None:
    debug_name = model.debug_names.names.get(id)
    if debug_name is not None:
        return debug_name
    if definition is None:
        return None
    for operand in definition.instruction.operands:
        if operand.kind == "IdResult" and operand.name is not None:
            return operand.name
    return None


def format_type(
    type_model: Type | None, model: SpirvModuleModel | None
) -> str:
    if type_model is None:
        return ""
    if isinstance(type_model, ScalarType):
        return format_element_type(type_model)
    if isinstance(type_model, TensorArmType):
        if model is None:
            return "tensor"
        return tensor_shape_metadata_value(type_model) or "tensor"
    if isinstance(type_model, GraphArmType):
        inputs = ", ".join(
            format_type(input_type, model)
            for input_type in type_model.input_types
        )
        outputs = ", ".join(
            format_type(output_type, model)
            for output_type in type_model.output_types
        )
        return f"({inputs}) -> ({outputs})"
    if isinstance(type_model, ArrayType):
        length = format_constant_value(type_model.length)
        element = format_type(type_model.element_type, model)
        return f"array<{length} x {element}>"
    if isinstance(type_model, RuntimeArrayType):
        element = format_type(type_model.element_type, model)
        return f"runtime_array<{element}>"
    return type_model.opcode


def format_constant_value(constant: Constant | None) -> str:
    if constant is None:
        return ""
    return format_literal_value(constant.resolved_value)


def format_constant_operand_value(
    constant: Constant,
    model: SpirvModuleModel,
    *,
    integer_interpretation: IntegerInterpretation = INTEGER_INTERPRETATION_TYPE,
) -> str:
    value = constant_display_value(constant, model, integer_interpretation)
    type_model = constant.type
    if isinstance(type_model, TensorArmType):
        return format_tensor_constant_value(
            value, type_model, integer_interpretation=integer_interpretation
        )
    if isinstance(type_model, VectorType | ArrayType):
        formatted_element_type = format_element_type(
            element_type(type_model),
            integer_interpretation=integer_interpretation,
        )
        return format_typed_sequence_constant_value(
            formatted_element_type, value
        )
    if isinstance(type_model, MatrixType | StructType):
        return format_sequence_literal(value)
    if isinstance(type_model, ScalarType) and type_model.kind in {
        "int",
        "float",
    }:
        formatted_element_type = format_element_type(
            type_model, integer_interpretation=integer_interpretation
        )
        return f"{formatted_element_type}[{format_literal_value(value)}]"
    return format_literal_value(value)


def element_type(type_model: Type | None) -> Type | None:
    if isinstance(type_model, VectorType):
        return type_model.component_type
    if isinstance(type_model, ArrayType):
        return type_model.element_type
    return None


def constant_display_value(
    constant: Constant,
    model: SpirvModuleModel,
    integer_interpretation: IntegerInterpretation,
) -> object:
    # The semantic model stores constants after type-based decoding. For TOSA
    # display overrides, that decoded value may be the wrong view of the same
    # raw bits (for example, uint32 0xffffffff should display as int32 -1 for a
    # signless TOSA offset). Re-decode integer constants from the original raw
    # SPIR-V words when an override is requested.
    if integer_interpretation == INTEGER_INTERPRETATION_TYPE:
        return constant.resolved_value
    type_model = constant.type
    if isinstance(type_model, TensorArmType):
        return reinterpret_tensor_constant_value(
            constant, type_model, model, integer_interpretation
        )
    if not isinstance(type_model, ScalarType) or type_model.kind != "int":
        return constant.resolved_value
    return decode_integer_constant_value(
        constant, type_model, integer_interpretation
    )


def reinterpret_tensor_constant_value(
    constant: Constant,
    type_model: TensorArmType,
    model: SpirvModuleModel,
    integer_interpretation: IntegerInterpretation,
) -> object:
    # Tensor constants are composites of scalar constants. Reinterpret the
    # referenced scalar constants rather than the already-resolved tensor tuple,
    # so signed/unsigned display overrides apply to every tensor element.
    if not isinstance(type_model.element_type, ScalarType):
        return constant.resolved_value
    if type_model.element_type.kind != "int":
        return constant.resolved_value
    if constant.opcode not in {
        "OpConstantComposite",
        "OpSpecConstantComposite",
        "OpConstantCompositeReplicateEXT",
    }:
        return constant.resolved_value
    values = tuple(
        reinterpret_constant_reference(
            constant_id,
            model,
            integer_interpretation,
        )
        for constant_id in constant.value
    )
    if constant.opcode != "OpConstantCompositeReplicateEXT":
        return values
    if not is_sequence(constant.resolved_value):
        return values
    count = len(constant.resolved_value)
    return tuple(values[0] for _index in range(count)) if values else ()


def reinterpret_constant_reference(
    constant_id: object,
    model: SpirvModuleModel,
    integer_interpretation: IntegerInterpretation,
) -> object:
    if not isinstance(constant_id, int) or constant_id not in model.constants:
        return constant_id
    referenced_constant = model.constants[constant_id]
    referenced_type = referenced_constant.type
    if (
        not isinstance(referenced_type, ScalarType)
        or referenced_type.kind != "int"
    ):
        return referenced_constant.resolved_value
    return decode_integer_constant_value(
        referenced_constant, referenced_type, integer_interpretation
    )


def decode_integer_constant_value(
    constant: Constant,
    type_model: ScalarType,
    integer_interpretation: IntegerInterpretation,
) -> object:
    width = type_model.width
    words = constant.instruction.raw_words[3:]
    if width is None or width <= 0 or not words:
        return constant.resolved_value
    word_count = (width + 31) // 32
    if len(words) < word_count:
        return constant.resolved_value
    value = 0
    for index, word in enumerate(words[:word_count]):
        value |= (word & 0xFFFFFFFF) << (index * 32)
    value &= (1 << width) - 1
    if integer_interpretation == INTEGER_INTERPRETATION_SIGNED and value & (
        1 << (width - 1)
    ):
        value -= 1 << width
    return value


def format_tensor_constant_value(
    value: object,
    type_model: TensorArmType,
    *,
    integer_interpretation: IntegerInterpretation = INTEGER_INTERPRETATION_TYPE,
) -> str:
    element_type = format_element_type(
        type_model.element_type,
        integer_interpretation=integer_interpretation,
    )
    dimensions = tensor_shape_dimensions(type_model)
    if is_sequence(value):
        return format_typed_sequence_constant_value(element_type, value)
    if dimensions:
        return f"{element_type}[{format_literal_value(value)}]"
    return f"{element_type} {format_literal_value(value)}"


def format_literal_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def format_typed_sequence_constant_value(
    element_type: str, value: object
) -> str:
    if not is_sequence(value):
        return f"{element_type}{format_sequence_literal(value)}"
    length = len(value)
    if length <= CONSTANT_SEQUENCE_PREVIEW_LENGTH:
        return f"{element_type}{format_sequence_literal(value)}"
    if is_zero_sequence(value):
        return f"{element_type}[<all zeroes>] len={length}"
    preview = value[:CONSTANT_SEQUENCE_PREVIEW_LENGTH]
    return (
        f"{element_type}["
        + ", ".join(format_sequence_literal(item) for item in preview)
        + f", ...] len={length}"
    )


def is_zero_sequence(value: object) -> bool:
    return is_sequence(value) and all(is_zero_value(item) for item in value)


def is_zero_value(value: object) -> bool:
    if is_sequence(value):
        return is_zero_sequence(value)
    return value == 0 or value is False


def format_sequence_literal(value: object) -> str:
    if not is_sequence(value):
        return format_literal_value(value)
    return (
        "[" + ", ".join(format_sequence_literal(item) for item in value) + "]"
    )


def is_sequence(value: object) -> TypeGuard[tuple[object, ...]]:
    return isinstance(value, tuple)
