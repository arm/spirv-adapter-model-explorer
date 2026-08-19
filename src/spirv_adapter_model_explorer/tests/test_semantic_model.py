# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from spirv_adapter_model_explorer.spirv.constants import (
    _resolve_null_constant_value,
)
from spirv_adapter_model_explorer.spirv.semantic_model import (
    build_semantic_model,
)
from spirv_adapter_model_explorer.spirv.spirv_grammar import load_core_grammar
from spirv_adapter_model_explorer.spirv.spirv_parser import (
    SPIRV_MAGIC,
    SpirvHeader,
    SpirvInstruction,
    SpirvModule,
    SpirvOperand,
    parse_spirv,
)
from spirv_adapter_model_explorer.spirv.types import (
    ArrayType,
    Constant,
    DataGraph,
    DataGraphEntryPoint,
    DataGraphInput,
    DataGraphOperation,
    DataGraphOutput,
    DataGraphType,
    FunctionType,
    MatrixType,
    PointerType,
    ScalarType,
    StructType,
    TensorType,
    VectorType,
)

from .conftest import fixture_text
from .test_spirv_parser import assemble_invalid_spirv

SEMANTIC_ASM = fixture_text("semantic-model.spvasm")


def _semantic_model():
    data = assemble_invalid_spirv(
        SEMANTIC_ASM,
        expected_diagnostic="Opcode TypeTensorARM requires one of these capabilities",
    )
    module = parse_spirv(data, load_core_grammar())
    return build_semantic_model(module, load_core_grammar())


def _named_id(model, name: str) -> int:
    return next(
        id
        for id, id_name in model.debug_names.names.items()
        if id_name == name
    )


def _type_id(model, opcode: str) -> int:
    return next(
        id
        for id, type_model in model.types.items()
        if type_model.opcode == opcode
    )


def test_collects_module_level_metadata():
    model = _semantic_model()

    assert model.capabilities == ("Shader",)
    assert model.extensions == ("SPV_KHR_storage_buffer_storage_class",)
    assert model.memory_model is not None
    assert model.memory_model.addressing_model == "Logical"
    assert model.memory_model.memory_model == "GLSL450"
    assert model.extended_instruction_imports[1].name == "TOSA.001000.1"


def test_collects_definitions_and_uses():
    model = _semantic_model()
    function_id = _named_id(model, "main_func")
    parameter_id = next(iter(model.functions[function_id].parameters)).id
    first_block_id = model.functions[function_id].blocks[0].id
    variable_id = _named_id(model, "buffer")

    assert model.definitions[function_id].kind == "function"
    assert (
        model.definitions[function_id].type_id
        == model.functions[function_id].return_type_id
    )
    assert model.definitions[parameter_id].kind == "function_parameter"
    assert model.definitions[first_block_id].kind == "basic_block"
    assert model.definitions[variable_id].kind == "variable"
    assert [
        use.user_instruction.opname for use in model.uses[variable_id]
    ] == [
        "OpEntryPoint",
        "OpName",
        "OpDecorate",
        "OpDecorate",
        "OpGroupDecorate",
    ]


def test_collects_debug_names():
    model = _semantic_model()
    function_id = _named_id(model, "main_func")
    struct_id = _named_id(model, "Buffer")

    assert model.debug_names.display_name(function_id) == "main_func"
    assert model.debug_names.display_name(999) == "%999"
    assert model.debug_names.member_display_name(struct_id, 0) == "value"
    assert model.debug_names.member_display_name(struct_id, 2) == "2"


def test_collects_decorations_and_decoration_groups():
    model = _semantic_model()
    variable_id = _named_id(model, "buffer")
    struct_id = _named_id(model, "Buffer")

    assert [
        decoration.decoration
        for decoration in model.decorations.by_target[variable_id]
    ] == [
        "DescriptorSet",
        "Binding",
        "Restrict",
    ]
    assert [
        decoration.operands
        for decoration in model.decorations.by_target[variable_id]
    ] == [
        (0,),
        (1,),
        (),
    ]
    assert (
        model.decorations.by_member[(struct_id, 0)][0].decoration == "Offset"
    )
    assert model.decorations.by_member[(struct_id, 0)][0].operands == (0,)


def test_collects_types():
    model = _semantic_model()
    void_id = _type_id(model, "OpTypeVoid")
    bool_id = _type_id(model, "OpTypeBool")
    int_id = _type_id(model, "OpTypeInt")
    float_id = _type_id(model, "OpTypeFloat")
    vector_id = _type_id(model, "OpTypeVector")
    array_id = _type_id(model, "OpTypeArray")
    storage_pointer_id = model.variables[_named_id(model, "buffer")].type_id
    struct_id = _named_id(model, "Buffer")
    function_type_id = next(iter(model.functions.values())).function_type_id

    assert model.types[void_id] == ScalarType(
        void_id, "OpTypeVoid", model.types[void_id].instruction, "void"
    )
    assert model.types[bool_id] == ScalarType(
        bool_id, "OpTypeBool", model.types[bool_id].instruction, "bool"
    )
    assert model.types[int_id] == ScalarType(
        int_id,
        "OpTypeInt",
        model.types[int_id].instruction,
        "int",
        width=32,
        signed=True,
    )
    assert model.types[float_id] == ScalarType(
        float_id,
        "OpTypeFloat",
        model.types[float_id].instruction,
        "float",
        width=32,
    )
    assert model.types[vector_id] == VectorType(
        vector_id,
        "OpTypeVector",
        model.types[vector_id].instruction,
        float_id,
        4,
    )
    assert isinstance(model.types[array_id], ArrayType)
    assert model.types[array_id].element_type_id == int_id
    assert model.types[storage_pointer_id] == PointerType(
        storage_pointer_id,
        "OpTypePointer",
        model.types[storage_pointer_id].instruction,
        "StorageBuffer",
        struct_id,
    )
    assert model.types[struct_id] == StructType(
        struct_id,
        "OpTypeStruct",
        model.types[struct_id].instruction,
        (int_id,),
    )
    assert model.types[function_type_id] == FunctionType(
        function_type_id,
        "OpTypeFunction",
        model.types[function_type_id].instruction,
        void_id,
        (int_id,),
    )

    tensor_id = _type_id(model, "OpTypeTensorARM")
    tensor = model.types[tensor_id]
    assert tensor == TensorType(
        tensor_id,
        "OpTypeTensorARM",
        tensor.instruction,
        element_type_id=int_id,
        rank_id=model.types[array_id].length_id,
        shape_id=next(
            constant.id
            for constant in model.constants.values()
            if constant.type_id == array_id
        ),
    )

    graph_id = _type_id(model, "OpTypeGraphARM")
    graph = model.types[graph_id]
    assert graph == DataGraphType(
        graph_id,
        "OpTypeGraphARM",
        graph.instruction,
        input_type_ids=(tensor_id,),
        output_type_ids=(tensor_id,),
    )


def test_collects_constants():
    model = _semantic_model()
    true_constant = next(
        constant
        for constant in model.constants.values()
        if constant.value is True
    )
    false_constant = next(
        constant
        for constant in model.constants.values()
        if constant.value is False
    )
    literal_42 = next(
        constant
        for constant in model.constants.values()
        if constant.value == 42
    )
    composites = [
        constant
        for constant in model.constants.values()
        if constant.opcode == "OpConstantComposite"
    ]
    composite = composites[0]
    nested_composite = composites[1]

    assert true_constant.opcode == "OpConstantTrue"
    assert true_constant.resolved_value is True
    assert false_constant.opcode == "OpConstantFalse"
    assert false_constant.resolved_value is False
    assert literal_42.type_id == _type_id(model, "OpTypeInt")
    assert literal_42.resolved_value == 42
    assert composite.value == (
        literal_42.id,
        literal_42.id,
        literal_42.id,
        literal_42.id,
    )
    assert composite.resolved_value == (42, 42, 42, 42)
    assert nested_composite.resolved_value == (
        (42, 42, 42, 42),
        (42, 42, 42, 42),
        (42, 42, 42, 42),
        (42, 42, 42, 42),
    )


@pytest.mark.parametrize(
    ("type_operands", "literal_words", "expected_value"),
    [
        (("OpTypeInt", 32, 1), (0xFFFFFFFF,), -1),
        (("OpTypeInt", 8, 1), (0x80,), -128),
        (("OpTypeInt", 8, 0), (0x80,), 128),
        (("OpTypeInt", 64, 1), (0xFFFFFFFF, 0xFFFFFFFF), -1),
        (("OpTypeInt", 64, 0), (0xFFFFFFFF, 0xFFFFFFFF), 18446744073709551615),
        (("OpTypeFloat", 16), (0x3800,), 0.5),
        (("OpTypeFloat", 32), (0x3F000000,), 0.5),
        (("OpTypeFloat", 64), (0x00000000, 0x3FE00000), 0.5),
    ],
)
def test_decodes_context_dependent_constant_literals_by_result_type(
    type_operands, literal_words, expected_value
):
    def literal(value: int) -> SpirvOperand:
        return SpirvOperand(kind="LiteralInteger", display=str(value))

    type_opcode, *type_operand_values = type_operands
    type_operands_tuple = (
        SpirvOperand(kind="IdResult", display="%1"),
    ) + tuple(literal(value) for value in type_operand_values)
    constant_operands = (
        SpirvOperand(kind="IdResultType", display="%1"),
        SpirvOperand(kind="IdResult", display="%2"),
    ) + tuple(
        SpirvOperand(kind="LiteralContextDependentNumber", display=str(word))
        for word in literal_words
    )
    module = SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=3,
            schema=0,
        ),
        instructions=(
            SpirvInstruction(
                index=0,
                word_offset=5,
                word_count=1 + len(type_operands_tuple),
                opcode=0,
                opname=type_opcode,
                operands=type_operands_tuple,
                raw_words=(),
                result_id=1,
            ),
            SpirvInstruction(
                index=1,
                word_offset=9,
                word_count=3 + len(literal_words),
                opcode=43,
                opname="OpConstant",
                operands=constant_operands,
                raw_words=(
                    ((3 + len(literal_words)) << 16) | 43,
                    1,
                    2,
                    *literal_words,
                ),
                result_type_id=1,
                result_id=2,
            ),
        ),
    )

    model = build_semantic_model(module, load_core_grammar())

    assert model.constants[2].value == expected_value
    assert model.constants[2].resolved_value == expected_value


def test_resolves_replicated_constant_values():
    def id_result(id: int) -> SpirvOperand:
        return SpirvOperand(kind="IdResult", display=f"%{id}")

    def id_ref(id: int) -> SpirvOperand:
        return SpirvOperand(kind="IdRef", display=f"%{id}", ref_id=id)

    def literal(value: int) -> SpirvOperand:
        return SpirvOperand(kind="LiteralInteger", display=str(value))

    def inst(
        index: int,
        opname: str,
        operands: tuple[SpirvOperand, ...],
        *,
        result_id: int | None = None,
        result_type_id: int | None = None,
        raw_words: tuple[int, ...] = (),
    ) -> SpirvInstruction:
        return SpirvInstruction(
            index=index,
            word_offset=index,
            word_count=1 + len(operands),
            opcode=0,
            opname=opname,
            operands=operands,
            raw_words=raw_words,
            result_id=result_id,
            result_type_id=result_type_id,
        )

    module = SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=20,
            schema=0,
        ),
        instructions=(
            inst(
                0,
                "OpTypeInt",
                (id_result(1), literal(32), literal(0)),
                result_id=1,
            ),
            inst(
                1,
                "OpConstant",
                (
                    SpirvOperand(kind="IdResultType", display="%1"),
                    id_result(2),
                    SpirvOperand(
                        kind="LiteralContextDependentNumber", display="3"
                    ),
                ),
                result_type_id=1,
                result_id=2,
                raw_words=(262187, 1, 2, 3),
            ),
            inst(
                2,
                "OpTypeVector",
                (id_result(3), id_ref(1), literal(4)),
                result_id=3,
            ),
            inst(
                3,
                "OpConstantCompositeReplicateEXT",
                (
                    SpirvOperand(kind="IdResultType", display="%3"),
                    id_result(4),
                    id_ref(2),
                ),
                result_type_id=3,
                result_id=4,
            ),
            inst(
                4,
                "OpConstant",
                (
                    SpirvOperand(kind="IdResultType", display="%1"),
                    id_result(5),
                    SpirvOperand(
                        kind="LiteralContextDependentNumber", display="2"
                    ),
                ),
                result_type_id=1,
                result_id=5,
                raw_words=(262187, 1, 5, 2),
            ),
            inst(
                5,
                "OpTypeArray",
                (id_result(6), id_ref(1), id_ref(5)),
                result_id=6,
            ),
            inst(
                6,
                "OpConstantCompositeReplicateEXT",
                (
                    SpirvOperand(kind="IdResultType", display="%6"),
                    id_result(7),
                    id_ref(2),
                ),
                result_type_id=6,
                result_id=7,
            ),
            inst(
                7,
                "OpTypeMatrix",
                (id_result(8), id_ref(3), literal(2)),
                result_id=8,
            ),
            inst(
                8,
                "OpConstantCompositeReplicateEXT",
                (
                    SpirvOperand(kind="IdResultType", display="%8"),
                    id_result(9),
                    id_ref(4),
                ),
                result_type_id=8,
                result_id=9,
            ),
            inst(
                9,
                "OpTypeStruct",
                (id_result(10), id_ref(1), id_ref(1), id_ref(1)),
                result_id=10,
            ),
            inst(
                10,
                "OpConstantCompositeReplicateEXT",
                (
                    SpirvOperand(kind="IdResultType", display="%10"),
                    id_result(11),
                    id_ref(2),
                ),
                result_type_id=10,
                result_id=11,
            ),
        ),
    )

    model = build_semantic_model(module, load_core_grammar())

    assert model.constants[4].value == (2,)
    assert model.constants[4].resolved_value == (3, 3, 3, 3)
    assert model.constants[7].resolved_value == (3, 3)
    assert model.constants[9].resolved_value == (
        (3, 3, 3, 3),
        (3, 3, 3, 3),
    )
    assert model.constants[11].resolved_value == (3, 3, 3)


def test_resolve_null_constant_value_uses_type_defaults():
    instruction = SpirvInstruction(
        index=0,
        word_offset=0,
        word_count=1,
        opcode=0,
        opname="test",
        operands=(),
        raw_words=(),
    )
    bool_type = ScalarType(1, "OpTypeBool", instruction, "bool")
    uint8_type = ScalarType(
        2, "OpTypeInt", instruction, "int", width=8, signed=False
    )
    float_type = ScalarType(3, "OpTypeFloat", instruction, "float", width=32)
    vector_type = VectorType(
        4,
        "OpTypeVector",
        instruction,
        component_type_id=2,
        component_count=4,
        component_type=uint8_type,
    )
    length = Constant(
        5,
        2,
        "OpConstant",
        2,
        instruction,
        type=uint8_type,
        resolved_value=2,
    )
    array_type = ArrayType(
        6,
        "OpTypeArray",
        instruction,
        element_type_id=2,
        length_id=5,
        element_type=uint8_type,
        length=length,
    )
    matrix_type = MatrixType(
        7,
        "OpTypeMatrix",
        instruction,
        column_type_id=4,
        column_count=2,
        column_type=vector_type,
    )
    struct_type = StructType(
        8,
        "OpTypeStruct",
        instruction,
        member_type_ids=(1, 2, 3),
        member_types=(bool_type, uint8_type, float_type),
    )
    pointer_type = PointerType(
        9,
        "OpTypePointer",
        instruction,
        "UniformConstant",
        pointee_type_id=2,
        pointee_type=uint8_type,
    )
    shape = Constant(
        10,
        2,
        "OpConstantComposite",
        (2, 2),
        instruction,
        type=array_type,
        resolved_value=(2, 2),
    )
    tensor_type = TensorType(
        11,
        "OpTypeTensorARM",
        instruction,
        element_type_id=2,
        rank_id=5,
        shape_id=10,
        element_type=uint8_type,
        rank=length,
        shape=shape,
    )

    def resolve(constant):
        return constant.resolved_value

    assert _resolve_null_constant_value(bool_type, resolve) is False
    assert _resolve_null_constant_value(uint8_type, resolve) == 0
    assert _resolve_null_constant_value(float_type, resolve) == 0.0
    assert _resolve_null_constant_value(vector_type, resolve) == (0, 0, 0, 0)
    assert _resolve_null_constant_value(array_type, resolve) == (0, 0)
    assert _resolve_null_constant_value(matrix_type, resolve) == (
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    assert _resolve_null_constant_value(struct_type, resolve) == (
        False,
        0,
        0.0,
    )
    assert _resolve_null_constant_value(pointer_type, resolve) is None
    assert _resolve_null_constant_value(tensor_type, resolve) == (0, 0, 0, 0)


def test_collects_variables():
    model = _semantic_model()

    variable = model.variables[_named_id(model, "buffer")]
    assert model.types[variable.type_id].opcode == "OpTypePointer"
    assert variable.storage_class == "StorageBuffer"
    assert variable.initializer_id is None


def test_collects_entry_points_and_execution_modes():
    model = _semantic_model()
    function_id = _named_id(model, "main_func")
    variable_id = _named_id(model, "buffer")

    entry_point = model.entry_points[function_id]
    assert entry_point.execution_model == "GLCompute"
    assert entry_point.name == "main"
    assert entry_point.interface_ids == (variable_id,)
    assert model.execution_modes[0].entry_point_id == function_id
    assert model.execution_modes[0].mode == "LocalSize"
    assert model.execution_modes[0].operands == (1, 2, 3)


def test_collects_functions_blocks_and_cfg_successors():
    model = _semantic_model()
    function = model.functions[_named_id(model, "main_func")]

    assert model.types[function.return_type_id].opcode == "OpTypeVoid"
    assert model.types[function.function_type_id].opcode == "OpTypeFunction"
    assert function.control == "0x00000000"
    assert [(param.id, param.type_id) for param in function.parameters] == [
        (function.parameters[0].id, _type_id(model, "OpTypeInt"))
    ]
    assert len(function.blocks) == 4
    assert [len(block.successors) for block in function.blocks] == [2, 1, 1, 0]
    assert function.blocks[0].successors == (
        function.blocks[1].id,
        function.blocks[2].id,
    )
    assert function.blocks[1].successors == (function.blocks[3].id,)
    assert function.blocks[2].successors == (function.blocks[3].id,)
    assert function.blocks[0].terminator is not None
    assert function.blocks[0].terminator.opname == "OpBranchConditional"
    assert function.blocks[-1].terminator is not None
    assert function.blocks[-1].terminator.opname == "OpReturn"


def test_preserves_global_instructions_outside_functions():
    model = _semantic_model()

    assert model.global_instructions[-1].opname == "OpTypeFunction"
    assert "OpFunction" not in [
        inst.opname for inst in model.global_instructions
    ]


def test_resolves_rich_references_for_valid_ids():
    model = _semantic_model()
    int_id = _type_id(model, "OpTypeInt")
    array_id = _type_id(model, "OpTypeArray")
    struct_id = _named_id(model, "Buffer")
    function_id = _named_id(model, "main_func")
    variable_id = _named_id(model, "buffer")
    tensor_id = _type_id(model, "OpTypeTensorARM")
    graph_id = _type_id(model, "OpTypeGraphARM")

    array = model.types[array_id]
    assert isinstance(array, ArrayType)
    assert array.element_type is model.types[int_id]
    assert array.length is model.constants[array.length_id]

    struct = model.types[struct_id]
    assert isinstance(struct, StructType)
    assert struct.member_types == (model.types[int_id],)

    variable = model.variables[variable_id]
    assert variable.type is model.types[variable.type_id]

    tensor = model.types[tensor_id]
    assert isinstance(tensor, TensorType)
    assert tensor.element_type is model.types[int_id]
    assert tensor.rank is model.constants[tensor.rank_id]
    assert tensor.shape is model.constants[tensor.shape_id]

    graph = model.types[graph_id]
    assert isinstance(graph, DataGraphType)
    assert graph.input_types == (tensor,)
    assert graph.output_types == (tensor,)

    function = model.functions[function_id]
    assert function.return_type is model.types[function.return_type_id]
    assert function.function_type is model.types[function.function_type_id]
    assert (
        function.parameters[0].type
        is model.types[function.parameters[0].type_id]
    )
    assert function.blocks[0].successor_blocks == (
        function.blocks[1],
        function.blocks[2],
    )

    entry_point = model.entry_points[function_id]
    assert entry_point.function is function
    assert entry_point.interfaces == (variable,)
    assert model.execution_modes[0].entry_point is entry_point


def test_resolved_reference_tuples_preserve_positional_none_alignment():
    module = SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=1000,
            schema=0,
        ),
        instructions=(
            SpirvInstruction(
                index=0,
                word_offset=5,
                word_count=4,
                opcode=21,
                opname="OpTypeInt",
                operands=(
                    SpirvOperand(kind="IdResult", display="%1"),
                    SpirvOperand(kind="LiteralInteger", display="32"),
                    SpirvOperand(kind="LiteralInteger", display="0"),
                ),
                raw_words=(),
                result_id=1,
            ),
            SpirvInstruction(
                index=1,
                word_offset=9,
                word_count=4,
                opcode=21,
                opname="OpTypeInt",
                operands=(
                    SpirvOperand(kind="IdResult", display="%2"),
                    SpirvOperand(kind="LiteralInteger", display="16"),
                    SpirvOperand(kind="LiteralInteger", display="0"),
                ),
                raw_words=(),
                result_id=2,
            ),
            SpirvInstruction(
                index=2,
                word_offset=13,
                word_count=5,
                opcode=30,
                opname="OpTypeStruct",
                operands=(
                    SpirvOperand(kind="IdResult", display="%3"),
                    SpirvOperand(kind="IdRef", display="%1", ref_id=1),
                    SpirvOperand(kind="IdRef", display="%999", ref_id=999),
                    SpirvOperand(kind="IdRef", display="%2", ref_id=2),
                ),
                raw_words=(),
                result_id=3,
            ),
            SpirvInstruction(
                index=3,
                word_offset=18,
                word_count=5,
                opcode=4190,
                opname="OpTypeGraphARM",
                operands=(
                    SpirvOperand(kind="IdResult", display="%4"),
                    SpirvOperand(kind="LiteralInteger", display="2"),
                    SpirvOperand(kind="IdRef", display="%3", ref_id=3),
                    SpirvOperand(kind="IdRef", display="%999", ref_id=999),
                    SpirvOperand(kind="IdRef", display="%1", ref_id=1),
                ),
                raw_words=(),
                result_id=4,
            ),
        ),
    )

    model = build_semantic_model(module, load_core_grammar())

    struct = model.types[3]
    assert isinstance(struct, StructType)
    assert struct.member_type_ids == (1, 999, 2)
    assert struct.member_types == (model.types[1], None, model.types[2])

    graph = model.types[4]
    assert isinstance(graph, DataGraphType)
    assert graph.input_type_ids == (3, 999)
    assert graph.output_type_ids == (1,)
    assert graph.input_types == (struct, None)
    assert graph.output_types == (model.types[1],)


def test_collects_data_graph_entry_points_and_resolved_interfaces():
    module = SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=40,
            schema=0,
        ),
        instructions=(
            SpirvInstruction(
                index=0,
                word_offset=5,
                word_count=5,
                opcode=4190,
                opname="OpTypeGraphARM",
                operands=(
                    SpirvOperand(kind="IdResult", display="%10"),
                    SpirvOperand(kind="LiteralInteger", display="1"),
                    SpirvOperand(kind="IdRef", display="%20", ref_id=20),
                    SpirvOperand(kind="IdRef", display="%21", ref_id=21),
                ),
                raw_words=(),
                result_id=10,
            ),
            SpirvInstruction(
                index=1,
                word_offset=10,
                word_count=3,
                opcode=4183,
                opname="OpGraphARM",
                operands=(
                    SpirvOperand(kind="IdResultType", display="%10"),
                    SpirvOperand(kind="IdResult", display="%11"),
                ),
                raw_words=(),
                result_type_id=10,
                result_id=11,
            ),
            SpirvInstruction(
                index=2,
                word_offset=13,
                word_count=4,
                opcode=59,
                opname="OpVariable",
                operands=(
                    SpirvOperand(kind="IdResultType", display="%20"),
                    SpirvOperand(kind="IdResult", display="%2"),
                    SpirvOperand(
                        kind="StorageClass", display="UniformConstant(0)"
                    ),
                ),
                raw_words=(),
                result_type_id=20,
                result_id=2,
            ),
            SpirvInstruction(
                index=3,
                word_offset=17,
                word_count=4,
                opcode=59,
                opname="OpVariable",
                operands=(
                    SpirvOperand(kind="IdResultType", display="%21"),
                    SpirvOperand(kind="IdResult", display="%3"),
                    SpirvOperand(
                        kind="StorageClass", display="UniformConstant(0)"
                    ),
                ),
                raw_words=(),
                result_type_id=21,
                result_id=3,
            ),
            SpirvInstruction(
                index=4,
                word_offset=21,
                word_count=6,
                opcode=4182,
                opname="OpGraphEntryPointARM",
                operands=(
                    SpirvOperand(kind="IdRef", display="%11", ref_id=11),
                    SpirvOperand(kind="LiteralString", display='"main"'),
                    SpirvOperand(kind="IdRef", display="%2", ref_id=2),
                    SpirvOperand(kind="IdRef", display="%3", ref_id=3),
                ),
                raw_words=(),
            ),
        ),
    )

    model = build_semantic_model(module, load_core_grammar())

    assert model.entry_points == {}
    assert len(model.data_graph_entry_points) == 1
    entry_point = model.data_graph_entry_points[0]
    assert isinstance(entry_point, DataGraphEntryPoint)
    assert entry_point.graph_id == 11
    assert entry_point.name == "main"
    assert entry_point.interface_ids == (2, 3)
    assert entry_point.graph_definition is model.definitions[11]
    assert entry_point.graph_definition.kind == "graph"
    assert entry_point.graph is model.data_graphs[11]
    assert entry_point.graph_type is model.types[10]
    assert isinstance(entry_point.graph_type, DataGraphType)
    assert entry_point.interfaces == (model.variables[2], model.variables[3])


def test_collects_data_graph_body_inputs_operations_and_outputs():
    module = SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=40,
            schema=0,
        ),
        instructions=(
            SpirvInstruction(
                index=0,
                word_offset=5,
                word_count=6,
                opcode=11,
                opname="OpExtInstImport",
                operands=(
                    SpirvOperand(kind="IdResult", display="%1"),
                    SpirvOperand(
                        kind="LiteralString", display='"TOSA.001000.1"'
                    ),
                ),
                raw_words=(),
                result_id=1,
            ),
            SpirvInstruction(
                index=1,
                word_offset=11,
                word_count=4,
                opcode=21,
                opname="OpTypeInt",
                operands=(
                    SpirvOperand(kind="IdResult", display="%20"),
                    SpirvOperand(kind="LiteralInteger", display="8"),
                    SpirvOperand(kind="LiteralInteger", display="0"),
                ),
                raw_words=(),
                result_id=20,
            ),
            SpirvInstruction(
                index=2,
                word_offset=15,
                word_count=4,
                opcode=21,
                opname="OpTypeInt",
                operands=(
                    SpirvOperand(kind="IdResult", display="%21"),
                    SpirvOperand(kind="LiteralInteger", display="32"),
                    SpirvOperand(kind="LiteralInteger", display="0"),
                ),
                raw_words=(),
                result_id=21,
            ),
            SpirvInstruction(
                index=3,
                word_offset=19,
                word_count=4,
                opcode=43,
                opname="OpConstant",
                operands=(
                    SpirvOperand(kind="IdResultType", display="%21"),
                    SpirvOperand(kind="IdResult", display="%6"),
                    SpirvOperand(
                        kind="LiteralContextDependentNumber", display="0"
                    ),
                ),
                raw_words=(262187, 21, 6, 0),
                result_type_id=21,
                result_id=6,
            ),
            SpirvInstruction(
                index=4,
                word_offset=23,
                word_count=4,
                opcode=43,
                opname="OpConstant",
                operands=(
                    SpirvOperand(kind="IdResultType", display="%21"),
                    SpirvOperand(kind="IdResult", display="%7"),
                    SpirvOperand(
                        kind="LiteralContextDependentNumber", display="1"
                    ),
                ),
                raw_words=(262187, 21, 7, 1),
                result_type_id=21,
                result_id=7,
            ),
            SpirvInstruction(
                index=5,
                word_offset=27,
                word_count=5,
                opcode=4190,
                opname="OpTypeGraphARM",
                operands=(
                    SpirvOperand(kind="IdResult", display="%10"),
                    SpirvOperand(kind="LiteralInteger", display="1"),
                    SpirvOperand(kind="IdRef", display="%20", ref_id=20),
                    SpirvOperand(kind="IdRef", display="%21", ref_id=21),
                ),
                raw_words=(),
                result_id=10,
            ),
            SpirvInstruction(
                index=6,
                word_offset=32,
                word_count=3,
                opcode=4183,
                opname="OpGraphARM",
                operands=(
                    SpirvOperand(kind="IdResultType", display="%10"),
                    SpirvOperand(kind="IdResult", display="%11"),
                ),
                raw_words=(),
                result_type_id=10,
                result_id=11,
            ),
            SpirvInstruction(
                index=7,
                word_offset=35,
                word_count=4,
                opcode=4184,
                opname="OpGraphInputARM",
                operands=(
                    SpirvOperand(kind="IdResultType", display="%20"),
                    SpirvOperand(kind="IdResult", display="%30"),
                    SpirvOperand(kind="IdRef", display="%6", ref_id=6),
                ),
                raw_words=(),
                result_type_id=20,
                result_id=30,
            ),
            SpirvInstruction(
                index=8,
                word_offset=39,
                word_count=7,
                opcode=12,
                opname="OpExtInst",
                operands=(
                    SpirvOperand(kind="IdResultType", display="%21"),
                    SpirvOperand(kind="IdResult", display="%31"),
                    SpirvOperand(kind="IdRef", display="%1", ref_id=1),
                    SpirvOperand(kind="LiteralExtInstInteger", display="9"),
                    SpirvOperand(kind="IdRef", display="%30", ref_id=30),
                    SpirvOperand(kind="IdRef", display="%7", ref_id=7),
                ),
                raw_words=(),
                result_type_id=21,
                result_id=31,
            ),
            SpirvInstruction(
                index=9,
                word_offset=45,
                word_count=3,
                opcode=4185,
                opname="OpGraphSetOutputARM",
                operands=(
                    SpirvOperand(kind="IdRef", display="%31", ref_id=31),
                    SpirvOperand(kind="IdRef", display="%7", ref_id=7),
                ),
                raw_words=(),
            ),
            SpirvInstruction(
                index=10,
                word_offset=48,
                word_count=1,
                opcode=4186,
                opname="OpGraphEndARM",
                operands=(),
                raw_words=(),
            ),
        ),
    )

    model = build_semantic_model(module, load_core_grammar())

    graph = model.data_graphs[11]
    assert isinstance(graph, DataGraph)
    assert graph.id == 11
    assert graph.type_id == 10
    assert graph.type is model.types[10]
    assert [inst.opname for inst in graph.body_instructions] == [
        "OpGraphInputARM",
        "OpExtInst",
        "OpGraphSetOutputARM",
        "OpGraphEndARM",
    ]

    assert graph.inputs == (
        DataGraphInput(
            id=30,
            type_id=20,
            input_index_id=6,
            instruction=graph.inputs[0].instruction,
        ),
    )
    assert graph.inputs[0].type is model.types[20]
    assert graph.inputs[0].input_index is model.constants[6]
    assert graph.inputs[0].element_index_id is None
    assert graph.inputs[0].element_index is None

    assert graph.operations == (
        DataGraphOperation(
            id=31,
            type_id=21,
            opcode_name="TRANSPOSE_CONV2D",
            operand_ids=(30, 7),
            instruction=graph.operations[0].instruction,
            ext_inst_set_id=1,
            ext_inst_opcode=9,
            ext_inst_set_name="TOSA.001000.1",
        ),
    )
    assert graph.operations[0].type is model.types[21]
    assert (
        graph.operations[0].ext_inst_set
        is model.extended_instruction_imports[1]
    )
    assert graph.operations[0].ext_inst_spec is not None
    assert graph.operations[0].ext_inst_spec.opname == "TRANSPOSE_CONV2D"
    assert tuple(
        operand.name for operand in graph.operations[0].ext_inst_spec.operands
    ) == (
        "out_pad",
        "stride",
        "acc_type",
        "local_bound",
        "input",
        "weight",
        "bias",
        "input_zp",
        "weight_zp",
    )
    assert graph.operations[0].operands == (model.definitions[30], 1)

    assert graph.outputs == (
        DataGraphOutput(
            value_id=31,
            output_index_id=7,
            instruction=graph.outputs[0].instruction,
        ),
    )
    assert graph.outputs[0].value is model.definitions[31]
    assert graph.outputs[0].output_index is model.constants[7]
    assert graph.outputs[0].element_index_id is None
    assert graph.outputs[0].element_index is None


def test_collects_data_graph_input_output_element_indexes():
    module = SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=40,
            schema=0,
        ),
        instructions=(
            SpirvInstruction(
                index=0,
                word_offset=5,
                word_count=4,
                opcode=21,
                opname="OpTypeInt",
                operands=(
                    SpirvOperand(kind="IdResult", display="%20"),
                    SpirvOperand(kind="LiteralInteger", display="32"),
                    SpirvOperand(kind="LiteralInteger", display="0"),
                ),
                raw_words=(),
                result_id=20,
            ),
            SpirvInstruction(
                index=1,
                word_offset=9,
                word_count=4,
                opcode=43,
                opname="OpConstant",
                operands=(
                    SpirvOperand(kind="IdResultType", display="%20"),
                    SpirvOperand(kind="IdResult", display="%6"),
                    SpirvOperand(
                        kind="LiteralContextDependentNumber", display="0"
                    ),
                ),
                raw_words=(262187, 20, 6, 0),
                result_type_id=20,
                result_id=6,
            ),
            SpirvInstruction(
                index=2,
                word_offset=13,
                word_count=4,
                opcode=43,
                opname="OpConstant",
                operands=(
                    SpirvOperand(kind="IdResultType", display="%20"),
                    SpirvOperand(kind="IdResult", display="%7"),
                    SpirvOperand(
                        kind="LiteralContextDependentNumber", display="1"
                    ),
                ),
                raw_words=(262187, 20, 7, 1),
                result_type_id=20,
                result_id=7,
            ),
            SpirvInstruction(
                index=3,
                word_offset=17,
                word_count=5,
                opcode=4190,
                opname="OpTypeGraphARM",
                operands=(
                    SpirvOperand(kind="IdResult", display="%10"),
                    SpirvOperand(kind="LiteralInteger", display="1"),
                    SpirvOperand(kind="IdRef", display="%20", ref_id=20),
                    SpirvOperand(kind="IdRef", display="%20", ref_id=20),
                ),
                raw_words=(),
                result_id=10,
            ),
            SpirvInstruction(
                index=4,
                word_offset=22,
                word_count=3,
                opcode=4183,
                opname="OpGraphARM",
                operands=(
                    SpirvOperand(kind="IdResultType", display="%10"),
                    SpirvOperand(kind="IdResult", display="%11"),
                ),
                raw_words=(),
                result_type_id=10,
                result_id=11,
            ),
            SpirvInstruction(
                index=5,
                word_offset=25,
                word_count=5,
                opcode=4184,
                opname="OpGraphInputARM",
                operands=(
                    SpirvOperand(kind="IdResultType", display="%20"),
                    SpirvOperand(kind="IdResult", display="%30"),
                    SpirvOperand(kind="IdRef", display="%6", ref_id=6),
                    SpirvOperand(kind="IdRef", display="%7", ref_id=7),
                ),
                raw_words=(),
                result_type_id=20,
                result_id=30,
            ),
            SpirvInstruction(
                index=6,
                word_offset=30,
                word_count=4,
                opcode=4185,
                opname="OpGraphSetOutputARM",
                operands=(
                    SpirvOperand(kind="IdRef", display="%30", ref_id=30),
                    SpirvOperand(kind="IdRef", display="%6", ref_id=6),
                    SpirvOperand(kind="IdRef", display="%7", ref_id=7),
                ),
                raw_words=(),
            ),
            SpirvInstruction(
                index=7,
                word_offset=34,
                word_count=1,
                opcode=4186,
                opname="OpGraphEndARM",
                operands=(),
                raw_words=(),
            ),
        ),
    )

    model = build_semantic_model(module, load_core_grammar())
    graph = model.data_graphs[11]

    assert graph.inputs[0].element_index_id == 7
    assert graph.inputs[0].element_index is model.constants[7]
    assert graph.outputs[0].element_index_id == 7
    assert graph.outputs[0].element_index is model.constants[7]


def test_graph_operation_ext_inst_decoding_fallbacks():
    module = SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=1000,
            schema=0,
        ),
        instructions=(
            SpirvInstruction(
                index=0,
                word_offset=5,
                word_count=6,
                opcode=11,
                opname="OpExtInstImport",
                operands=(
                    SpirvOperand(kind="IdResult", display="%1"),
                    SpirvOperand(
                        kind="LiteralString", display='"TOSA.001000.1"'
                    ),
                ),
                raw_words=(),
                result_id=1,
            ),
            SpirvInstruction(
                index=1,
                word_offset=11,
                word_count=4,
                opcode=21,
                opname="OpTypeInt",
                operands=(
                    SpirvOperand(kind="IdResult", display="%20"),
                    SpirvOperand(kind="LiteralInteger", display="32"),
                    SpirvOperand(kind="LiteralInteger", display="0"),
                ),
                raw_words=(),
                result_id=20,
            ),
            SpirvInstruction(
                index=2,
                word_offset=15,
                word_count=4,
                opcode=4190,
                opname="OpTypeGraphARM",
                operands=(
                    SpirvOperand(kind="IdResult", display="%10"),
                    SpirvOperand(kind="LiteralInteger", display="0"),
                    SpirvOperand(kind="IdRef", display="%20", ref_id=20),
                ),
                raw_words=(),
                result_id=10,
            ),
            SpirvInstruction(
                index=3,
                word_offset=19,
                word_count=3,
                opcode=4183,
                opname="OpGraphARM",
                operands=(
                    SpirvOperand(kind="IdResultType", display="%10"),
                    SpirvOperand(kind="IdResult", display="%11"),
                ),
                raw_words=(),
                result_type_id=10,
                result_id=11,
            ),
            SpirvInstruction(
                index=4,
                word_offset=22,
                word_count=4,
                opcode=12,
                opname="OpExtInst",
                operands=(
                    SpirvOperand(kind="IdResultType", display="%20"),
                    SpirvOperand(kind="IdResult", display="%30"),
                    SpirvOperand(kind="IdRef", display="%1", ref_id=1),
                    SpirvOperand(kind="LiteralExtInstInteger", display="999"),
                ),
                raw_words=(),
                result_type_id=20,
                result_id=30,
            ),
            SpirvInstruction(
                index=5,
                word_offset=26,
                word_count=4,
                opcode=12,
                opname="OpExtInst",
                operands=(
                    SpirvOperand(kind="IdResultType", display="%20"),
                    SpirvOperand(kind="IdResult", display="%31"),
                    SpirvOperand(kind="IdRef", display="%999", ref_id=999),
                    SpirvOperand(kind="LiteralExtInstInteger", display="1"),
                ),
                raw_words=(),
                result_type_id=20,
                result_id=31,
            ),
            SpirvInstruction(
                index=6,
                word_offset=30,
                word_count=1,
                opcode=4186,
                opname="OpGraphEndARM",
                operands=(),
                raw_words=(),
            ),
        ),
    )

    model = build_semantic_model(module, load_core_grammar())
    unknown_opcode, unknown_set = model.data_graphs[11].operations

    assert unknown_opcode.ext_inst_set_name == "TOSA.001000.1"
    assert unknown_opcode.ext_inst_opcode == 999
    assert unknown_opcode.ext_inst_spec is None
    assert unknown_opcode.opcode_name == "OpExtInst 999"

    assert unknown_set.ext_inst_set_name is None
    assert unknown_set.ext_inst_opcode == 1
    assert unknown_set.ext_inst_spec is None
    assert unknown_set.opcode_name == "OpExtInst 1"
