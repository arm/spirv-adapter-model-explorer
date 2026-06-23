# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .spirv_grammar import InstructionSpec
from .spirv_parser import SpirvHeader, SpirvInstruction

Id = int


@dataclass(frozen=True)
class Definition:
    id: Id
    instruction: SpirvInstruction
    kind: str
    type_id: Id | None


@dataclass(frozen=True)
class Use:
    used_id: Id
    user_instruction: SpirvInstruction
    operand_index: int
    role: str | None


@dataclass(frozen=True)
class DebugNames:
    names: dict[Id, str] = field(default_factory=dict)
    member_names: dict[tuple[Id, int], str] = field(default_factory=dict)

    def display_name(self, id: Id) -> str:
        return self.names.get(id, f"%{id}")

    def member_display_name(self, id: Id, member_index: int) -> str:
        return self.member_names.get((id, member_index), str(member_index))


@dataclass(frozen=True)
class Decoration:
    target_id: Id
    decoration: str
    operands: tuple[Any, ...]
    instruction: SpirvInstruction


@dataclass(frozen=True)
class MemberDecoration:
    target_id: Id
    member_index: int
    decoration: str
    operands: tuple[Any, ...]
    instruction: SpirvInstruction


@dataclass(frozen=True)
class DecorationIndex:
    by_target: dict[Id, tuple[Decoration, ...]] = field(default_factory=dict)
    by_member: dict[tuple[Id, int], tuple[MemberDecoration, ...]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class Type:
    id: Id
    opcode: str
    instruction: SpirvInstruction


@dataclass(frozen=True)
class ScalarType(Type):
    kind: Literal["void", "bool", "int", "float"]
    width: int | None = None
    signed: bool | None = None


@dataclass(frozen=True)
class VectorType(Type):
    component_type_id: Id
    component_count: int
    component_type: Type | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class MatrixType(Type):
    column_type_id: Id
    column_count: int
    column_type: Type | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class PointerType(Type):
    storage_class: str
    pointee_type_id: Id
    pointee_type: Type | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ArrayType(Type):
    element_type_id: Id
    length_id: Id
    element_type: Type | None = field(default=None, repr=False, compare=False)
    length: Constant | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class RuntimeArrayType(Type):
    element_type_id: Id
    element_type: Type | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class StructType(Type):
    member_type_ids: tuple[Id, ...]
    member_types: tuple[Type | None, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )


@dataclass(frozen=True)
class FunctionType(Type):
    return_type_id: Id
    parameter_type_ids: tuple[Id, ...]
    return_type: Type | None = field(default=None, repr=False, compare=False)
    parameter_types: tuple[Type | None, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )


@dataclass(frozen=True)
class TensorArmType(Type):
    element_type_id: Id
    rank_id: Id
    shape_id: Id | None = None
    element_type: Type | None = field(default=None, repr=False, compare=False)
    rank: Constant | None = field(default=None, repr=False, compare=False)
    shape: Constant | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class GraphArmType(Type):
    input_type_ids: tuple[Id, ...]
    output_type_ids: tuple[Id, ...]
    input_types: tuple[Type | None, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )
    output_types: tuple[Type | None, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )


@dataclass(frozen=True)
class UnknownType(Type):
    operands: tuple[Any, ...]


@dataclass(frozen=True)
class Constant:
    id: Id
    type_id: Id
    opcode: str
    value: Any
    instruction: SpirvInstruction
    type: Type | None = field(default=None, repr=False, compare=False)
    resolved_value: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class Variable:
    id: Id
    type_id: Id
    storage_class: str
    initializer_id: Id | None
    instruction: SpirvInstruction
    type: Type | None = field(default=None, repr=False, compare=False)
    initializer: Constant | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class ExtInstImport:
    id: Id
    name: str
    instruction: SpirvInstruction


@dataclass(frozen=True)
class MemoryModel:
    addressing_model: str
    memory_model: str
    instruction: SpirvInstruction


@dataclass(frozen=True)
class EntryPoint:
    id: Id
    execution_model: str
    name: str
    interface_ids: tuple[Id, ...]
    instruction: SpirvInstruction
    function: Function | None = field(default=None, repr=False, compare=False)
    interfaces: tuple[Variable | None, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )


@dataclass(frozen=True)
class GraphArmInput:
    id: Id
    type_id: Id
    input_index_id: Id
    instruction: SpirvInstruction
    element_index_id: Id | None = None
    debug_info: tuple[str, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )
    type: Type | None = field(default=None, repr=False, compare=False)
    input_index: Constant | None = field(
        default=None, repr=False, compare=False
    )
    element_index: Constant | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class GraphArmOperation:
    id: Id
    type_id: Id
    opcode_name: str
    operand_ids: tuple[Id, ...]
    instruction: SpirvInstruction
    ext_inst_set_id: Id | None = None
    ext_inst_opcode: int | None = None
    ext_inst_set_name: str | None = None
    debug_info: tuple[str, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )
    output_debug_info: tuple[tuple[str, ...], ...] = field(
        default_factory=tuple, repr=False, compare=False
    )
    ext_inst_spec: InstructionSpec | None = field(
        default=None, repr=False, compare=False
    )
    type: Type | None = field(default=None, repr=False, compare=False)
    ext_inst_set: ExtInstImport | None = field(
        default=None, repr=False, compare=False
    )
    # Semantic view of ``operand_ids``, with the same ordering and arity.
    # Each entry is one of:
    # - the embedded SPIR-V constant's ``resolved_value``;
    # - a ``Definition`` for non-constant IDs known to the module; or
    # - ``None`` for unknown IDs.
    # Note that OpConstantNull also resolves to ``None``; callers that need
    # to distinguish that from an unknown ID should check the corresponding
    # raw ID in ``SpirvModuleModel.constants``.
    operands: tuple[object, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )


@dataclass(frozen=True)
class GraphArmOutput:
    value_id: Id
    output_index_id: Id
    instruction: SpirvInstruction
    element_index_id: Id | None = None
    debug_info: tuple[str, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )
    value: Definition | None = field(default=None, repr=False, compare=False)
    output_index: Constant | None = field(
        default=None, repr=False, compare=False
    )
    element_index: Constant | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class GraphArm:
    id: Id
    type_id: Id
    instruction: SpirvInstruction
    body_instructions: tuple[SpirvInstruction, ...]
    inputs: tuple[GraphArmInput, ...]
    operations: tuple[GraphArmOperation, ...]
    outputs: tuple[GraphArmOutput, ...]
    debug_info: tuple[str, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )
    type: GraphArmType | None = field(default=None, repr=False, compare=False)
    entry_points: tuple[GraphArmEntryPoint, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )


@dataclass(frozen=True)
class GraphArmEntryPoint:
    graph_id: Id
    name: str
    interface_ids: tuple[Id, ...]
    instruction: SpirvInstruction
    graph_definition: Definition | None = field(
        default=None, repr=False, compare=False
    )
    graph: GraphArm | None = field(default=None, repr=False, compare=False)
    graph_type: GraphArmType | None = field(
        default=None, repr=False, compare=False
    )
    interfaces: tuple[Variable | None, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )


@dataclass(frozen=True)
class ExecutionMode:
    entry_point_id: Id
    mode: str
    operands: tuple[Any, ...]
    instruction: SpirvInstruction
    entry_point: EntryPoint | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class FunctionParameter:
    id: Id
    type_id: Id
    instruction: SpirvInstruction
    type: Type | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class BasicBlock:
    id: Id
    label: SpirvInstruction
    instructions: tuple[SpirvInstruction, ...]
    terminator: SpirvInstruction | None
    successors: tuple[Id, ...]
    successor_blocks: tuple[BasicBlock | None, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )


@dataclass(frozen=True)
class Function:
    id: Id
    return_type_id: Id
    function_type_id: Id
    control: str
    parameters: tuple[FunctionParameter, ...]
    blocks: tuple[BasicBlock, ...]
    instruction: SpirvInstruction
    return_type: Type | None = field(default=None, repr=False, compare=False)
    function_type: FunctionType | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class SpirvModuleModel:
    header: SpirvHeader
    capabilities: tuple[str, ...]
    extensions: tuple[str, ...]
    extended_instruction_imports: dict[Id, ExtInstImport]
    memory_model: MemoryModel | None
    entry_points: dict[Id, EntryPoint]
    graph_entry_points: tuple[GraphArmEntryPoint, ...]
    execution_modes: tuple[ExecutionMode, ...]
    debug_names: DebugNames
    decorations: DecorationIndex
    types: dict[Id, Type]
    constants: dict[Id, Constant]
    variables: dict[Id, Variable]
    functions: dict[Id, Function]
    graphs: dict[Id, GraphArm]
    global_instructions: tuple[SpirvInstruction, ...]
    definitions: dict[Id, Definition]
    uses: dict[Id, tuple[Use, ...]]


def _set_resolved_attr(obj: object, name: str, value: object) -> None:
    object.__setattr__(obj, name, value)


def _semantic_operands_after_result(inst: SpirvInstruction) -> tuple[Any, ...]:
    return tuple(
        _operand_value(operand)
        for operand in inst.operands
        if operand.kind not in {"IdResult", "IdResultType"}
    )


def _operand_value(operand) -> Any:
    if operand.ref_id is not None:
        return operand.ref_id
    if operand.kind == "LiteralString":
        return _literal_string(operand.display)
    literal = _literal_int(operand.display)
    if literal is not None:
        return literal
    return _enum_display_name(operand.display)


def _literal_int(display: str) -> int | None:
    try:
        return int(display, 0)
    except ValueError:
        return None


def _literal_string(display: str) -> str:
    if len(display) >= 2 and display[0] == '"' and display[-1] == '"':
        return display[1:-1]
    return display


def _enum_display_name(display: str) -> str:
    if "(" in display and display.endswith(")"):
        return display[: display.rfind("(")]
    return display


__all__ = [
    "ArrayType",
    "BasicBlock",
    "Constant",
    "DebugNames",
    "Decoration",
    "DecorationIndex",
    "Definition",
    "EntryPoint",
    "ExecutionMode",
    "ExtInstImport",
    "Function",
    "FunctionParameter",
    "FunctionType",
    "GraphArm",
    "GraphArmEntryPoint",
    "GraphArmInput",
    "GraphArmOperation",
    "GraphArmOutput",
    "GraphArmType",
    "Id",
    "MatrixType",
    "MemberDecoration",
    "MemoryModel",
    "PointerType",
    "RuntimeArrayType",
    "ScalarType",
    "SpirvModuleModel",
    "StructType",
    "TensorArmType",
    "Type",
    "UnknownType",
    "Use",
    "Variable",
    "VectorType",
]
