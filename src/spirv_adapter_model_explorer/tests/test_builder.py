# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from spirv_adapter_model_explorer.assembly import (
    SpirvDisassemblyLine,
    assemble_spirv_text,
)
from spirv_adapter_model_explorer.builder import (
    _composite_extract_selector,
    _incoming_edges_for_operation,
    _operation_output_types,
    _ProducerOutput,
    build_spirv_graphs_from_bytes,
    build_spirv_graphs_from_semantic_model,
)
from spirv_adapter_model_explorer.formatting import (
    format_constant_operand_value,
    format_constant_value,
    format_element_type,
    format_type,
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
)
from spirv_adapter_model_explorer.spirv.types import (
    ArrayType,
    Constant,
    RuntimeArrayType,
    ScalarType,
    TensorArmType,
    VectorType,
)
from spirv_adapter_model_explorer.tosa import operation_output_tags

from .test_spirv_parser import assemble_invalid_spirv


def _module_with_complex_arm_graph() -> SpirvModule:
    return SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=100,
            schema=0,
        ),
        instructions=(
            _ext_inst_import(0, 1, "TOSA.001000.1"),
            _type_int(1, 20, 8),
            _constant(2, 20, 6, 0),
            _constant(3, 20, 7, 1),
            _type_graph(4, 10, 2, (20, 20, 20)),
            _graph(5, 11, 10),
            _graph_input(6, 30, 20, 6),
            _graph_input(7, 31, 20, 7),
            _ext_inst(8, 40, 20, 1, 14, (30, 31)),
            _ext_inst(9, 41, 20, 1, 10, (40, 31)),
            _graph_output(10, 41, 6),
            _graph_end(11),
        ),
    )


def _module_with_constant_operand_arm_graph() -> SpirvModule:
    return SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=100,
            schema=0,
        ),
        instructions=(
            _ext_inst_import(0, 1, "TOSA.001000.1"),
            _type_int(1, 20, 8),
            _type_int(2, 21, 32),
            _constant(3, 21, 6, 0),
            _constant(4, 21, 7, 1),
            _constant(5, 21, 8, 4),
            _constant(6, 21, 9, 1),
            _type_array(7, 22, 21, 7),
            _constant_composite(8, 22, 23, (8,)),
            _type_tensor(9, 24, 20, 7, 23),
            _constant_composite_replicate(10, 24, 25, 9),
            _type_graph(11, 10, 1, (24, 24)),
            _graph(12, 11, 10),
            _graph_input(13, 30, 24, 6),
            _ext_inst(14, 40, 24, 1, 14, (30, 25)),
            _graph_output(15, 40, 6),
            _graph_end(16),
        ),
    )


def _module_with_graph_constant_operand_arm_graph() -> SpirvModule:
    return SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=100,
            schema=0,
        ),
        instructions=(
            _ext_inst_import(0, 1, "TOSA.001000.1"),
            _type_int(1, 20, 8),
            _type_int(2, 21, 32),
            _constant(3, 21, 6, 0),
            _constant(4, 21, 7, 1),
            _type_array(5, 22, 21, 7),
            _constant_composite(6, 22, 23, (7,)),
            _type_tensor(7, 24, 20, 7, 23),
            _type_graph(8, 10, 1, (24, 24)),
            _graph_constant(9, 31, 24, 14),
            _graph(10, 11, 10),
            _graph_input(11, 30, 24, 6),
            _ext_inst(12, 40, 24, 1, 14, (30, 31)),
            _graph_output(13, 40, 6),
            _graph_end(14),
        ),
    )


def _module_with_tosa_concat_arm_graph() -> SpirvModule:
    return SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=100,
            schema=0,
        ),
        instructions=(
            _ext_inst_import(0, 1, "TOSA.001000.1"),
            _type_int(1, 20, 8),
            _type_int(2, 21, 32),
            _constant(3, 21, 6, 0),
            _constant(4, 21, 7, 1),
            _constant(5, 21, 8, 2),
            _constant(6, 21, 9, 3),
            _type_array(7, 22, 21, 7),
            _constant_composite(8, 22, 23, (7,)),
            _type_tensor(9, 24, 20, 7, 23),
            _type_graph(10, 10, 3, (24, 24, 24, 24)),
            _graph(11, 11, 10),
            _name(11, 30, "input0"),
            _name(12, 31, "input1"),
            _name(13, 32, "input2"),
            _graph_input(14, 30, 24, 6),
            _graph_input(15, 31, 24, 7),
            _graph_input(16, 32, 24, 8),
            _ext_inst(17, 40, 24, 1, 54, (6, 30, 31, 32)),
            _graph_output(18, 40, 6),
            _graph_end(19),
        ),
    )


def _module_with_graph_debug_info_graph() -> SpirvModule:
    return SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=200,
            schema=0,
        ),
        instructions=(
            _ext_inst_import(0, 1, "TOSA.001000.1"),
            _ext_inst_import(1, 2, "NonSemantic.Graph.DebugInfo.1"),
            _ext_inst_import(2, 3, "NonSemantic.Shader.DebugInfo.100"),
            _string(3, 70, "graph debug"),
            _string(4, 71, "input debug"),
            _string(5, 72, "operation debug"),
            _string(6, 73, "real output debug"),
            _string(7, 74, "ignored shader debug"),
            _type_int(8, 20, 8),
            _type_int(9, 21, 32),
            _constant(10, 21, 6, 0),
            _constant(11, 21, 7, 1),
            _type_array(12, 22, 21, 7),
            _constant_composite(13, 22, 23, (7,)),
            _type_tensor(14, 24, 20, 7, 23),
            _type_struct(15, 25, (24, 24)),
            _type_graph(16, 10, 1, (24, 24, 24)),
            _graph(17, 11, 10),
            _graph_input(18, 30, 24, 6),
            _ext_inst(19, 40, 25, 1, 8, (30,)),
            _composite_extract(20, 41, 24, 40, (0,)),
            _composite_extract(21, 42, 24, 40, (1,)),
            _debug_ext_inst(22, 80, 2, 1, (11, 70)),
            _debug_ext_inst(23, 81, 2, 2, (80, 72, 40)),
            _debug_ext_inst(24, 82, 2, 3, (30, 71)),
            _debug_ext_inst(25, 83, 2, 3, (41, 73)),
            _debug_ext_inst(26, 84, 3, 2, (80, 74, 40)),
            _graph_output(27, 41, 6),
            _graph_output(28, 42, 7),
            _graph_end(29),
        ),
    )


def _module_with_tosa_rfft_struct_outputs_graph(
    *,
    real_extract_indexes: tuple[int, ...] = (0,),
    real_extract_type_id: int = 24,
    include_downstream_consumer: bool = False,
) -> SpirvModule:
    extra_instructions = (
        (_ext_inst(23, 43, 24, 1, 14, (41, 31)),)
        if include_downstream_consumer
        else ()
    )
    return SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=100,
            schema=0,
        ),
        instructions=(
            _ext_inst_import(0, 1, "TOSA.001000.1"),
            _type_int(1, 20, 8),
            _type_int(2, 21, 32),
            _constant(3, 21, 6, 0),
            _constant(4, 21, 7, 1),
            _type_array(5, 22, 21, 7),
            _constant_composite(6, 22, 23, (7,)),
            _type_tensor(7, 24, 20, 7, 23),
            _type_struct(8, 25, (24, 24)),
            _type_graph(9, 10, 2, (24, 24, 24, 24)),
            _graph(10, 11, 10),
            _name(11, 30, "local_bound"),
            _name(12, 31, "input_real"),
            _name(13, 40, "rfft"),
            _name(14, 41, "real"),
            _name(15, 42, "imag"),
            _graph_input(16, 30, 24, 6),
            _graph_input(17, 31, 24, 7),
            _ext_inst(18, 40, 25, 1, 8, (30, 31)),
            _composite_extract(
                19, 41, real_extract_type_id, 40, real_extract_indexes
            ),
            _composite_extract(20, 42, 24, 40, (1,)),
            *extra_instructions,
            _graph_output(21, 41, 6),
            _graph_output(22, 42, 7),
            _graph_end(24 if include_downstream_consumer else 23),
        ),
    )


def _module_with_composite_shape_tensor_graph() -> SpirvModule:
    return SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=100,
            schema=0,
        ),
        instructions=(
            _type_int(0, 20, 8),
            _type_int(1, 21, 32),
            _constant(2, 21, 6, 0),
            _constant(3, 21, 7, 4),
            _constant(4, 21, 8, 1),
            _type_array(5, 22, 21, 8),
            _constant_composite(6, 22, 23, (7,)),
            _type_tensor(7, 24, 20, 8, 23),
            _type_graph(8, 10, 1, (24, 24)),
            _graph(9, 11, 10),
            _graph_input(10, 30, 24, 6),
            _graph_output(11, 30, 6),
            _graph_end(12),
        ),
    )


def _module_with_tosa_signed_default_constant_operand() -> SpirvModule:
    return SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=100,
            schema=0,
        ),
        instructions=(
            _ext_inst_import(0, 1, "TOSA.001000.1"),
            _type_int(1, 20, 8),
            _type_int(2, 21, 32),
            _constant(3, 21, 6, 0),
            _constant(4, 21, 7, 1),
            _constant(5, 21, 8, 0xFFFFFFFF),
            _type_array(6, 22, 21, 7),
            _constant_composite(7, 22, 23, (8,)),
            _type_tensor(8, 24, 20, 7, 23),
            _type_graph(9, 10, 1, (24, 24)),
            _graph(10, 11, 10),
            _graph_input(11, 30, 24, 6),
            _ext_inst(12, 40, 24, 1, 14, (30, 8)),
            _graph_output(13, 40, 6),
            _graph_end(14),
        ),
    )


def _module_with_non_tosa_unsigned_constant_operand() -> SpirvModule:
    return SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=100,
            schema=0,
        ),
        instructions=(
            _ext_inst_import(0, 1, "Other.001"),
            _type_int(1, 20, 8),
            _type_int(2, 21, 32),
            _constant(3, 21, 6, 0),
            _constant(4, 21, 7, 1),
            _constant(5, 21, 8, 0xFFFFFFFF),
            _type_array(6, 22, 21, 7),
            _constant_composite(7, 22, 23, (8,)),
            _type_tensor(8, 24, 20, 7, 23),
            _type_graph(9, 10, 1, (24, 24)),
            _graph(10, 11, 10),
            _graph_input(11, 30, 24, 6),
            _ext_inst(12, 40, 24, 1, 999, (30, 8)),
            _graph_output(13, 40, 6),
            _graph_end(14),
        ),
    )


def _module_with_tosa_resize_signed_tensor_operand() -> SpirvModule:
    return SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=100,
            schema=0,
        ),
        instructions=(
            _ext_inst_import(0, 1, "TOSA.001000.1"),
            _type_int(1, 20, 8),
            _type_int(2, 21, 32),
            _constant(3, 21, 6, 0),
            _constant(4, 21, 7, 1),
            _constant(5, 21, 8, 2),
            _constant(6, 21, 9, 0xFFFFFFFF),
            _constant(7, 21, 17, 2),
            _type_array(8, 22, 21, 8),
            _constant_composite(9, 22, 24, (9, 9)),
            _type_tensor(10, 23, 21, 17, 24),
            _constant_composite(11, 23, 26, (9, 9)),
            _type_tensor(12, 25, 20, 7, 24),
            _type_graph(13, 10, 1, (25, 25)),
            _graph(14, 11, 10),
            _graph_input(15, 30, 25, 6),
            _ext_inst(16, 40, 25, 1, 63, (7, 30, 26, 26, 26)),
            _graph_output(17, 40, 6),
            _graph_end(18),
        ),
    )


def _module_with_tosa_rescale_constant_operands(
    *, input_unsigned: bool, output_unsigned: bool
) -> SpirvModule:
    return SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=100,
            schema=0,
        ),
        instructions=(
            _ext_inst_import(0, 1, "TOSA.001000.1"),
            _type_bool(1, 2),
            _type_int(2, 20, 8),
            _type_int(3, 21, 32),
            _constant(4, 21, 6, 0),
            _constant(5, 21, 7, 1),
            _constant_bool(6, 2, 8, True),
            _constant_bool(7, 2, 9, input_unsigned),
            _constant_bool(8, 2, 12, output_unsigned),
            _constant(9, 21, 13, 0xFFFFFFFF),
            _constant(10, 21, 14, 0xFFFFFFFF),
            _constant(11, 21, 15, 0xFFFFFFFF),
            _constant(12, 21, 16, 0xFFFFFFFF),
            _type_array(13, 22, 21, 7),
            _constant_composite(14, 22, 24, (13,)),
            _type_tensor(15, 23, 21, 7, 24),
            _type_tensor(16, 25, 20, 7, 24),
            _type_graph(17, 10, 1, (25, 25)),
            _graph(18, 11, 10),
            _graph_input(19, 30, 25, 6),
            _ext_inst(
                20,
                40,
                25,
                1,
                65,
                (8, 6, 8, 9, 12, 13, 14, 15, 16, 16),
            ),
            _graph_output(21, 40, 6),
            _graph_end(22),
        ),
    )


def _module_with_shape_dimensions_matching_constant_ids() -> SpirvModule:
    return SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=0,
            bound=100,
            schema=0,
        ),
        instructions=(
            _type_int(0, 1, 8),
            _type_int(1, 2, 32),
            _constant(2, 2, 3, 0),
            _constant(3, 2, 4, 20),
            _constant(4, 2, 5, 21),
            _constant(5, 2, 8, 2),
            _constant(6, 2, 20, 99),
            _constant(7, 2, 21, 100),
            _type_array(8, 6, 2, 8),
            _constant_composite(9, 6, 7, (4, 5)),
            _type_tensor(10, 9, 1, 8, 7),
            _type_graph(11, 10, 1, (9, 9)),
            _graph(12, 11, 10),
            _graph_input(13, 30, 9, 3),
            _graph_output(14, 30, 3),
            _graph_end(15),
        ),
    )


def _name(index: int, target_id: int, name: str) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=3,
        opcode=5,
        opname="OpName",
        operands=(
            SpirvOperand(
                kind="IdRef", display=f"%{target_id}", ref_id=target_id
            ),
            SpirvOperand(kind="LiteralString", display=f'"{name}"'),
        ),
        raw_words=(),
    )


def _ext_inst_import(
    index: int, result_id: int, name: str
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=2,
        opcode=11,
        opname="OpExtInstImport",
        operands=(
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
            SpirvOperand(kind="LiteralString", display=f'"{name}"'),
        ),
        raw_words=(),
        result_id=result_id,
    )


def _string(index: int, result_id: int, value: str) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=3,
        opcode=7,
        opname="OpString",
        operands=(
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
            SpirvOperand(kind="LiteralString", display=f'"{value}"'),
        ),
        raw_words=(),
        result_id=result_id,
    )


def _type_int(
    index: int, result_id: int, width: int, signedness: int = 0
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=4,
        opcode=21,
        opname="OpTypeInt",
        operands=(
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
            SpirvOperand(kind="LiteralInteger", display=str(width)),
            SpirvOperand(kind="LiteralInteger", display=str(signedness)),
        ),
        raw_words=(),
        result_id=result_id,
    )


def _type_bool(index: int, result_id: int) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=2,
        opcode=20,
        opname="OpTypeBool",
        operands=(SpirvOperand(kind="IdResult", display=f"%{result_id}"),),
        raw_words=(),
        result_id=result_id,
    )


def _constant(
    index: int, type_id: int, result_id: int, value: int, word_count: int = 4
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=word_count,
        opcode=43,
        opname="OpConstant",
        operands=(
            SpirvOperand(kind="IdResultType", display=f"%{type_id}"),
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
            SpirvOperand(
                kind="LiteralContextDependentNumber", display=str(value)
            ),
        ),
        raw_words=((word_count << 16) | 43, type_id, result_id, value),
        result_type_id=type_id,
        result_id=result_id,
    )


def _type_array(
    index: int, result_id: int, element_type_id: int, length_id: int
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=4,
        opcode=28,
        opname="OpTypeArray",
        operands=(
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
            SpirvOperand(
                kind="IdRef",
                display=f"%{element_type_id}",
                ref_id=element_type_id,
            ),
            SpirvOperand(
                kind="IdRef", display=f"%{length_id}", ref_id=length_id
            ),
        ),
        raw_words=(),
        result_id=result_id,
    )


def _type_runtime_array(
    index: int, result_id: int, element_type_id: int
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=3,
        opcode=29,
        opname="OpTypeRuntimeArray",
        operands=(
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
            SpirvOperand(
                kind="IdRef",
                display=f"%{element_type_id}",
                ref_id=element_type_id,
            ),
        ),
        raw_words=(),
        result_id=result_id,
    )


def _constant_bool(
    index: int, type_id: int, result_id: int, value: bool
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=3,
        opcode=41 if value else 42,
        opname="OpConstantTrue" if value else "OpConstantFalse",
        operands=(
            SpirvOperand(kind="IdResultType", display=f"%{type_id}"),
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
        ),
        raw_words=(),
        result_type_id=type_id,
        result_id=result_id,
    )


def _constant_composite(
    index: int, type_id: int, result_id: int, constituent_ids: tuple[int, ...]
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=3 + len(constituent_ids),
        opcode=44,
        opname="OpConstantComposite",
        operands=(
            SpirvOperand(kind="IdResultType", display=f"%{type_id}"),
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
            *tuple(
                SpirvOperand(kind="IdRef", display=f"%{id}", ref_id=id)
                for id in constituent_ids
            ),
        ),
        raw_words=(),
        result_type_id=type_id,
        result_id=result_id,
    )


def _constant_composite_replicate(
    index: int, type_id: int, result_id: int, constituent_id: int
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=4,
        opcode=4461,
        opname="OpConstantCompositeReplicateEXT",
        operands=(
            SpirvOperand(kind="IdResultType", display=f"%{type_id}"),
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
            SpirvOperand(
                kind="IdRef",
                display=f"%{constituent_id}",
                ref_id=constituent_id,
            ),
        ),
        raw_words=(),
        result_type_id=type_id,
        result_id=result_id,
    )


def _type_tensor(
    index: int,
    result_id: int,
    element_type_id: int,
    rank_id: int,
    shape_id: int,
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=5,
        opcode=4161,
        opname="OpTypeTensorARM",
        operands=(
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
            SpirvOperand(
                kind="IdRef",
                display=f"%{element_type_id}",
                ref_id=element_type_id,
            ),
            SpirvOperand(kind="IdRef", display=f"%{rank_id}", ref_id=rank_id),
            SpirvOperand(
                kind="IdRef", display=f"%{shape_id}", ref_id=shape_id
            ),
        ),
        raw_words=(),
        result_id=result_id,
    )


def _type_struct(
    index: int, result_id: int, member_type_ids: tuple[int, ...]
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=2 + len(member_type_ids),
        opcode=30,
        opname="OpTypeStruct",
        operands=(
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
            *tuple(
                SpirvOperand(kind="IdRef", display=f"%{id}", ref_id=id)
                for id in member_type_ids
            ),
        ),
        raw_words=(),
        result_id=result_id,
    )


def _type_graph(
    index: int,
    result_id: int,
    num_inputs: int,
    in_out_type_ids: tuple[int, ...],
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=3 + len(in_out_type_ids),
        opcode=4190,
        opname="OpTypeGraphARM",
        operands=(
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
            SpirvOperand(kind="LiteralInteger", display=str(num_inputs)),
            *tuple(
                SpirvOperand(kind="IdRef", display=f"%{id}", ref_id=id)
                for id in in_out_type_ids
            ),
        ),
        raw_words=(),
        result_id=result_id,
    )


def _graph(index: int, result_id: int, type_id: int) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=3,
        opcode=4183,
        opname="OpGraphARM",
        operands=(
            SpirvOperand(kind="IdResultType", display=f"%{type_id}"),
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
        ),
        raw_words=(),
        result_type_id=type_id,
        result_id=result_id,
    )


def _graph_input(
    index: int, result_id: int, type_id: int, input_index_id: int
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=4,
        opcode=4184,
        opname="OpGraphInputARM",
        operands=(
            SpirvOperand(kind="IdResultType", display=f"%{type_id}"),
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
            SpirvOperand(
                kind="IdRef",
                display=f"%{input_index_id}",
                ref_id=input_index_id,
            ),
        ),
        raw_words=(),
        result_type_id=type_id,
        result_id=result_id,
    )


def _graph_constant(
    index: int, result_id: int, type_id: int, graph_constant_id: int
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=4,
        opcode=4181,
        opname="OpGraphConstantARM",
        operands=(
            SpirvOperand(kind="IdResultType", display=f"%{type_id}"),
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
            SpirvOperand(
                kind="LiteralInteger", display=str(graph_constant_id)
            ),
        ),
        raw_words=(),
        result_type_id=type_id,
        result_id=result_id,
    )


def _debug_ext_inst(
    index: int,
    result_id: int,
    set_id: int,
    opcode: int,
    operand_ids: tuple[int, ...],
) -> SpirvInstruction:
    return _ext_inst(index, result_id, 20, set_id, opcode, operand_ids)


def _ext_inst(
    index: int,
    result_id: int,
    type_id: int,
    set_id: int,
    opcode: int,
    operand_ids: tuple[int, ...],
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=5 + len(operand_ids),
        opcode=12,
        opname="OpExtInst",
        operands=(
            SpirvOperand(kind="IdResultType", display=f"%{type_id}"),
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
            SpirvOperand(kind="IdRef", display=f"%{set_id}", ref_id=set_id),
            SpirvOperand(kind="LiteralExtInstInteger", display=str(opcode)),
            *tuple(
                SpirvOperand(kind="IdRef", display=f"%{id}", ref_id=id)
                for id in operand_ids
            ),
        ),
        raw_words=(),
        result_type_id=type_id,
        result_id=result_id,
    )


def _composite_extract(
    index: int,
    result_id: int,
    type_id: int,
    composite_id: int,
    indexes: tuple[int, ...],
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=4 + len(indexes),
        opcode=81,
        opname="OpCompositeExtract",
        operands=(
            SpirvOperand(kind="IdResultType", display=f"%{type_id}"),
            SpirvOperand(kind="IdResult", display=f"%{result_id}"),
            SpirvOperand(
                kind="IdRef", display=f"%{composite_id}", ref_id=composite_id
            ),
            *tuple(
                SpirvOperand(kind="LiteralInteger", display=str(index))
                for index in indexes
            ),
        ),
        raw_words=(),
        result_type_id=type_id,
        result_id=result_id,
    )


def _graph_output(
    index: int, value_id: int, output_index_id: int
) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=3,
        opcode=4185,
        opname="OpGraphSetOutputARM",
        operands=(
            SpirvOperand(
                kind="IdRef", display=f"%{value_id}", ref_id=value_id
            ),
            SpirvOperand(
                kind="IdRef",
                display=f"%{output_index_id}",
                ref_id=output_index_id,
            ),
        ),
        raw_words=(),
    )


def _graph_end(index: int) -> SpirvInstruction:
    return SpirvInstruction(
        index=index,
        word_offset=index,
        word_count=1,
        opcode=4186,
        opname="OpGraphEndARM",
        operands=(),
        raw_words=(),
    )


def _node_by_label(graph, label: str):
    return next(node for node in graph.nodes if node.label == label)


def _attr_value(node, key: str) -> str:
    return next(attr.value for attr in node.attrs if attr.key == key)


def _attrs(node) -> dict[str, str]:
    return {attr.key: attr.value for attr in node.attrs}


def _metadata_attrs(metadata_items):
    return {
        item.id: {attr.key: attr.value for attr in item.attrs}
        for item in metadata_items
    }


def test_format_element_type_uses_pytorch_like_type_names():
    instruction = _type_int(0, 1, 8)

    assert (
        format_element_type(ScalarType(1, "OpTypeBool", instruction, "bool"))
        == "bool"
    )
    assert (
        format_element_type(
            ScalarType(
                2, "OpTypeInt", instruction, "int", width=8, signed=False
            )
        )
        == "uint8"
    )
    assert (
        format_element_type(
            ScalarType(
                3, "OpTypeInt", instruction, "int", width=16, signed=True
            )
        )
        == "int16"
    )
    assert (
        format_element_type(
            ScalarType(4, "OpTypeFloat", instruction, "float", width=32)
        )
        == "float32"
    )


def test_formats_boolean_values_lowercase():
    bool_type_inst = _type_bool(0, 1)
    bool_type = ScalarType(1, "OpTypeBool", bool_type_inst, "bool")
    vector_type = VectorType(
        2, "OpTypeVector", bool_type_inst, 1, 2, component_type=bool_type
    )
    tensor_type = TensorArmType(
        3, "OpTypeTensorARM", bool_type_inst, 1, 4, element_type=bool_type
    )
    true_inst = _constant_bool(1, 1, 5, True)
    false_inst = _constant_bool(2, 1, 6, False)
    true_constant = Constant(
        5, 1, "OpConstantTrue", True, true_inst, bool_type, True
    )
    false_constant = Constant(
        6, 1, "OpConstantFalse", False, false_inst, bool_type, False
    )

    assert format_constant_value(true_constant) == "true"
    assert format_constant_operand_value(false_constant, None) == "false"
    assert (
        format_constant_operand_value(
            Constant(
                7,
                2,
                "OpConstantComposite",
                (5, 6),
                _constant_composite(3, 2, 7, (5, 6)),
                vector_type,
                (True, False),
            ),
            None,
        )
        == "bool[true, false]"
    )
    assert (
        format_constant_operand_value(
            Constant(
                8,
                3,
                "OpConstantComposite",
                (5, 6),
                _constant_composite(4, 3, 8, (5, 6)),
                tensor_type,
                (True, False),
            ),
            None,
        )
        == "bool[true, false]"
    )


def test_large_sequence_constant_operand_values_are_truncated():
    uint8_type_inst = _type_int(0, 1, 8)
    uint8_type = ScalarType(
        1, "OpTypeInt", uint8_type_inst, "int", width=8, signed=False
    )
    vector_type = VectorType(
        2,
        "OpTypeVector",
        uint8_type_inst,
        component_type_id=1,
        component_count=33,
        component_type=uint8_type,
    )
    expected_preview = ", ".join(str(value) for value in range(1, 33))

    assert (
        format_constant_operand_value(
            Constant(
                3,
                2,
                "OpConstantComposite",
                tuple(range(1, 34)),
                _constant_composite(1, 2, 3, tuple(range(1, 34))),
                vector_type,
                tuple(range(1, 34)),
            ),
            None,
        )
        == f"uint8[{expected_preview}, ...] len=33"
    )


def test_large_all_zero_sequence_constant_operand_values_are_summarized():
    uint8_type_inst = _type_int(0, 1, 8)
    uint8_type = ScalarType(
        1, "OpTypeInt", uint8_type_inst, "int", width=8, signed=False
    )
    vector_type = VectorType(
        2,
        "OpTypeVector",
        uint8_type_inst,
        component_type_id=1,
        component_count=33,
        component_type=uint8_type,
    )

    assert (
        format_constant_operand_value(
            Constant(
                3,
                2,
                "OpConstantComposite",
                tuple(0 for _index in range(33)),
                _constant_composite(1, 2, 3, tuple(range(33))),
                vector_type,
                tuple(0 for _index in range(33)),
            ),
            None,
        )
        == "uint8[<all zeroes>] len=33"
    )


def test_build_spirv_graphs_from_bytes_uses_raw_instruction_fallback():
    data = assemble_invalid_spirv(
        """
; SPIR-V
; Version: 1.0
; Bound: 6
; Schema: 0
OpCapability Shader
OpMemoryModel Logical GLSL450
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
OpReturn
OpFunctionEnd
""",
        expected_diagnostic="No OpEntryPoint instruction was found",
    )

    graphs = build_spirv_graphs_from_bytes(data)

    assert len(graphs) == 1
    assert graphs[0].id == "module"
    assert [node.label for node in graphs[0].nodes] == [
        "OpCapability",
        "OpMemoryModel",
        "OpTypeVoid",
        "OpTypeFunction",
        "OpFunction",
        "OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]


def test_builds_semantic_arm_graph_nodes_and_tosa_attrs():
    grammar = load_core_grammar()
    module = _module_with_complex_arm_graph()
    model = build_semantic_model(module, grammar)

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    assert [node.label for node in graph.nodes] == [
        "Input 0",
        "Input 1",
        "ADD",
        "CLAMP",
        "Output 0",
    ]
    add = _node_by_label(graph, "ADD")
    assert [attr.key for attr in add.attrs[:2]] == [
        "opcode",
        "result type",
    ]
    assert [attr.key for attr in add.attrs[-4:]] == [
        "spv id",
        "spv result type id",
        "spv ext set",
        "spv ext opcode",
    ]
    attr_keys = {attr.key for attr in add.attrs}
    assert "Ext inst set" not in attr_keys
    assert "Ext inst opcode" not in attr_keys
    assert _attr_value(add, "spv ext set") == "TOSA.001000.1"
    assert _attr_value(add, "spv ext opcode") == "14"
    assert _attr_value(add, "spec") == (
        "[TOSA spec: ADD]("
        "https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_add)"
    )
    assert "arg0 input1" not in attr_keys
    assert "arg1 input2" not in attr_keys
    assert "SPIR-V / operand 0 ID" not in attr_keys
    assert "SPIR-V / operand 1 ID" not in attr_keys


def test_builds_semantic_arm_graph_with_disassembly_source_attrs():
    grammar = load_core_grammar()
    module = _module_with_complex_arm_graph()
    model = build_semantic_model(module, grammar)
    lines_by_byte_offset = {
        inst.word_offset * 4: SpirvDisassemblyLine(
            line_number=inst.index + 1,
            byte_offset=inst.word_offset * 4,
            text=f"{inst.opname} ; 0x{inst.word_offset * 4:08x}",
        )
        for inst in module.instructions
    }

    graph = build_spirv_graphs_from_semantic_model(
        model,
        module=module,
        disassembly_text="full module disassembly",
        disassembly_lines_by_byte_offset=lines_by_byte_offset,
    )[0]
    input_node = _node_by_label(graph, "Input 0")
    add = _node_by_label(graph, "ADD")
    output = _node_by_label(graph, "Output 0")

    assert graph.groupNodeAttributes[""]["spv full asm"] == (
        "full module disassembly"
    )
    assert _attr_value(input_node, "spv byte offset") == "0x00000018"
    assert _attr_value(input_node, "spv asm") == (
        "OpGraphInputARM ; 0x00000018"
    )
    assert _attr_value(add, "spv byte offset") == "0x00000020"
    assert _attr_value(add, "spv asm") == ("OpExtInst ; 0x00000020")
    assert _attr_value(output, "spv byte offset") == "0x00000028"
    assert _attr_value(output, "spv asm") == (
        "OpGraphSetOutputARM ; 0x00000028"
    )


def test_builds_edges_for_complex_arm_graph_dependencies():
    grammar = load_core_grammar()
    module = _module_with_complex_arm_graph()
    model = build_semantic_model(module, grammar)

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    input_0 = _node_by_label(graph, "Input 0")
    input_1 = _node_by_label(graph, "Input 1")
    add = _node_by_label(graph, "ADD")
    clamp = _node_by_label(graph, "CLAMP")
    output = _node_by_label(graph, "Output 0")

    assert add.incomingEdges == [
        gb_edge(input_0.id, "0"),
        gb_edge(input_1.id, "1"),
    ]
    assert clamp.incomingEdges == [
        gb_edge(add.id, "0"),
        gb_edge(input_1.id, "1"),
    ]
    assert output.incomingEdges == [gb_edge(clamp.id, "0")]


def test_operation_attrs_show_resolved_constant_operand_values():
    grammar = load_core_grammar()
    module = _module_with_constant_operand_arm_graph()
    model = build_semantic_model(module, grammar)
    assert model.constants[25].resolved_value == (1, 1, 1, 1)

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    add = _node_by_label(graph, "ADD")
    attrs = {attr.key: attr.value for attr in add.attrs}
    assert "arg0 input1" not in attrs
    assert attrs["arg1 input2"] == "int8[1, 1, 1, 1]"
    assert "SPIR-V / operand 0 ID" not in attrs
    assert "SPIR-V / operand 1 ID" not in attrs


def test_operation_attrs_show_graph_constant_operand_ids():
    grammar = load_core_grammar()
    module = _module_with_graph_constant_operand_arm_graph()
    model = build_semantic_model(module, grammar)

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    add = _node_by_label(graph, "ADD")
    attrs = {attr.key: attr.value for attr in add.attrs}
    assert attrs["arg1 input2"] == "<graph constant 14> uint8[1]"


def test_graph_debug_info_strings_populate_graph_operation_and_port_attrs():
    grammar = load_core_grammar()
    module = _module_with_graph_debug_info_graph()
    model = build_semantic_model(module, grammar)
    semantic_graph = model.graphs[11]
    input0 = semantic_graph.inputs[0]
    rfft = next(
        operation
        for operation in semantic_graph.operations
        if operation.id == 40
    )

    assert semantic_graph.debug_info == ("graph debug",)
    assert input0.debug_info == ("input debug",)
    assert rfft.debug_info == ("operation debug",)
    assert rfft.output_debug_info == (("real output debug",),)
    assert "ignored shader debug" not in rfft.debug_info

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    graph_attrs = graph.groupNodeAttributes[""]
    input_node = _node_by_label(graph, "Input 0")
    rfft_node = _node_by_label(graph, "RFFT2D")
    output_node = _node_by_label(graph, "Output 0")
    assert graph_attrs["debug info"] == "graph debug"
    assert _attr_value(input_node, "debug info") == "input debug"
    assert _attr_value(rfft_node, "debug info") == "operation debug"
    assert (
        _metadata_attrs(rfft_node.outputsMetadata)["0"]["debug info"]
        == "real output debug"
    )
    assert _attr_value(output_node, "debug info") == "real output debug"


def test_tosa_repeated_operands_expand_attrs_and_input_tensor_tags():
    grammar = load_core_grammar()
    module = _module_with_tosa_concat_arm_graph()
    model = build_semantic_model(module, grammar)

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    concat = _node_by_label(graph, "CONCAT")
    attrs = {attr.key: attr.value for attr in concat.attrs}
    assert attrs["arg0 axis"] == "int32[0]"
    assert "arg1 input1" not in attrs
    assert "arg2 input2" not in attrs
    assert "arg3 input3" not in attrs
    assert concat.incomingEdges == [
        gb_edge(_node_by_label(graph, "Input 0").id, "1"),
        gb_edge(_node_by_label(graph, "Input 1").id, "2"),
        gb_edge(_node_by_label(graph, "Input 2").id, "3"),
    ]
    assert _metadata_attrs(concat.inputsMetadata) == {
        "0": {"__tensor_tag": "axis"},
        "1": {"__tensor_tag": "input1", "tensor_shape": "uint8[1]"},
        "2": {"__tensor_tag": "input2", "tensor_shape": "uint8[1]"},
        "3": {"__tensor_tag": "input3", "tensor_shape": "uint8[1]"},
    }


def test_operation_output_types_expands_struct_members():
    grammar = load_core_grammar()
    module = _module_with_tosa_rfft_struct_outputs_graph()
    model = build_semantic_model(module, grammar)
    graph = next(iter(model.graphs.values()))
    rfft = next(
        operation
        for operation in graph.operations
        if operation.opcode_name == "RFFT2D"
    )

    assert _operation_output_types(rfft) == rfft.type.member_types


def test_tosa_multi_output_tags_name_struct_result_ports():
    grammar = load_core_grammar()
    module = _module_with_tosa_rfft_struct_outputs_graph()
    model = build_semantic_model(module, grammar)
    graph = next(iter(model.graphs.values()))
    rfft = next(
        operation
        for operation in graph.operations
        if operation.opcode_name == "RFFT2D"
    )

    assert operation_output_tags(rfft) == ("output_real", "output_imag")


def test_composite_extract_selector_accepts_top_level_struct_member_extract():
    grammar = load_core_grammar()
    module = _module_with_tosa_rfft_struct_outputs_graph()
    model = build_semantic_model(module, grammar)
    graph = next(iter(model.graphs.values()))
    operations_by_id = {
        operation.id: operation for operation in graph.operations
    }
    real = operations_by_id[41]
    imag = operations_by_id[42]

    assert _composite_extract_selector(real, operations_by_id) == (40, "0")
    assert _composite_extract_selector(imag, operations_by_id) == (40, "1")


def test_composite_extract_selector_rejects_nested_extracts():
    grammar = load_core_grammar()
    module = _module_with_tosa_rfft_struct_outputs_graph(
        real_extract_indexes=(0, 0)
    )
    model = build_semantic_model(module, grammar)
    graph = next(iter(model.graphs.values()))
    operations_by_id = {
        operation.id: operation for operation in graph.operations
    }

    assert (
        _composite_extract_selector(operations_by_id[41], operations_by_id)
        is None
    )


def test_composite_extract_selector_rejects_type_mismatch():
    grammar = load_core_grammar()
    module = _module_with_tosa_rfft_struct_outputs_graph(
        real_extract_type_id=21
    )
    model = build_semantic_model(module, grammar)
    graph = next(iter(model.graphs.values()))
    operations_by_id = {
        operation.id: operation for operation in graph.operations
    }

    assert (
        _composite_extract_selector(operations_by_id[41], operations_by_id)
        is None
    )


def test_incoming_edges_for_operation_preserves_source_output_ports():
    grammar = load_core_grammar()
    module = _module_with_tosa_rfft_struct_outputs_graph(
        include_downstream_consumer=True
    )
    model = build_semantic_model(module, grammar)
    graph = next(iter(model.graphs.values()))
    consumer = next(
        operation for operation in graph.operations if operation.id == 43
    )

    assert _incoming_edges_for_operation(
        consumer, {41: _ProducerOutput("rfft", "0")}
    ) == [gb_edge("rfft", "0", source_output_id="0")]


def test_struct_operation_outputs_expand_to_member_ports():
    grammar = load_core_grammar()
    module = _module_with_tosa_rfft_struct_outputs_graph()
    model = build_semantic_model(module, grammar)

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    rfft = _node_by_label(graph, "RFFT2D")
    assert _metadata_attrs(rfft.outputsMetadata) == {
        "0": {"__tensor_tag": "output_real", "tensor_shape": "uint8[1]"},
        "1": {"__tensor_tag": "output_imag", "tensor_shape": "uint8[1]"},
    }


def test_composite_extracts_are_folded_into_struct_output_edges():
    grammar = load_core_grammar()
    module = _module_with_tosa_rfft_struct_outputs_graph()
    model = build_semantic_model(module, grammar)

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    assert [node.label for node in graph.nodes] == [
        "Input 0",
        "Input 1",
        "RFFT2D",
        "Output 0",
        "Output 1",
    ]
    rfft = _node_by_label(graph, "RFFT2D")
    output0 = _node_by_label(graph, "Output 0")
    output1 = _node_by_label(graph, "Output 1")
    rfft_attrs = {attr.key: attr.value for attr in rfft.attrs}
    assert "arg0 local_bound" not in rfft_attrs
    assert "arg1 input_real" not in rfft_attrs
    assert output0.incomingEdges == [
        gb_edge(rfft.id, "0", source_output_id="0")
    ]
    assert output1.incomingEdges == [
        gb_edge(rfft.id, "0", source_output_id="1")
    ]


def test_non_tosa_constant_operands_keep_spirv_unsigned_display():
    # The TOSA signless rules must not leak into generic SPIR-V display. For an
    # unknown extended instruction set, OpTypeInt 32 0 should still be shown with
    # the normal SPIR-V unsigned interpretation.
    grammar = load_core_grammar()
    module = _module_with_non_tosa_unsigned_constant_operand()
    model = build_semantic_model(module, grammar)

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    operation = _node_by_label(graph, "OpExtInst 999")
    attrs = {attr.key: attr.value for attr in operation.attrs}
    assert attrs["arg1"] == "uint32[4294967295]"
    assert "spec" not in attrs


def test_tosa_constant_operands_default_to_signed_signless_display():
    # TOSA i32 lowers to OpTypeInt 32 0, the same encoding as raw SPIR-V uint32.
    # TOSA semantics are signless/signed-by-default, so 0xffffffff should display
    # as -1 when consumed by a TOSA op.
    grammar = load_core_grammar()
    module = _module_with_tosa_signed_default_constant_operand()
    model = build_semantic_model(module, grammar)

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    add = _node_by_label(graph, "ADD")
    attrs = {attr.key: attr.value for attr in add.attrs}
    assert attrs["arg1 input2"] == "int32[-1]"


def test_tosa_tensor_constant_operands_default_to_signed_signless_display():
    # The NSS regression was a RESIZE offset tensor whose raw element was
    # 0xffffffff. This verifies that signed/signless reinterpretation applies to
    # composite tensor constants as well as scalar constants.
    grammar = load_core_grammar()
    module = _module_with_tosa_resize_signed_tensor_operand()
    model = build_semantic_model(module, grammar)

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    resize = _node_by_label(graph, "RESIZE")
    attrs = {attr.key: attr.value for attr in resize.attrs}
    assert attrs["arg2 scale"] == "int32[-1, -1]"
    assert attrs["arg3 offset"] == "int32[-1, -1]"
    assert attrs["arg4 border"] == "int32[-1, -1]"


def test_tosa_rescale_unsigned_flags_override_operand_display():
    # RESCALE is the TOSA exception to the signed-by-default rule. When its
    # unsigned flags are true, the corresponding data and zero-point operands
    # must display using unsigned interpretation, while unrelated integer
    # operands such as multiplier and shift remain signed/signless.
    grammar = load_core_grammar()
    module = _module_with_tosa_rescale_constant_operands(
        input_unsigned=True, output_unsigned=True
    )
    model = build_semantic_model(module, grammar)

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    rescale = _node_by_label(graph, "RESCALE")
    attrs = {attr.key: attr.value for attr in rescale.attrs}
    assert attrs["arg5 input"] == "uint32[4294967295]"
    assert attrs["arg8 input_zp"] == "uint32[4294967295]"
    assert attrs["arg9 output_zp"] == "uint32[4294967295]"
    assert attrs["arg6 multiplier"] == "int32[-1]"
    assert attrs["arg7 shift"] == "int32[-1]"
    assert attrs["arg0 scale32"] == "true"
    assert attrs["arg2 per_channel"] == "true"
    assert attrs["arg3 input_unsigned"] == "true"
    assert attrs["arg4 output_unsigned"] == "true"


def test_tosa_rescale_false_unsigned_flags_use_signed_display():
    # The presence of RESCALE alone is not enough to force unsigned display. The
    # flags have to be true; false flags fall back to normal TOSA signless/signed
    # interpretation.
    grammar = load_core_grammar()
    module = _module_with_tosa_rescale_constant_operands(
        input_unsigned=False, output_unsigned=False
    )
    model = build_semantic_model(module, grammar)

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    rescale = _node_by_label(graph, "RESCALE")
    attrs = {attr.key: attr.value for attr in rescale.attrs}
    assert attrs["arg3 input_unsigned"] == "false"
    assert attrs["arg4 output_unsigned"] == "false"
    assert attrs["arg5 input"] == "int32[-1]"
    assert attrs["arg8 input_zp"] == "int32[-1]"
    assert attrs["arg9 output_zp"] == "int32[-1]"


def test_tosa_rescale_input_and_output_unsigned_flags_are_independent():
    # input_unsigned and output_unsigned are separate controls. An unsigned input
    # tensor/zero-point must not make the output zero-point look unsigned too.
    grammar = load_core_grammar()
    module = _module_with_tosa_rescale_constant_operands(
        input_unsigned=True, output_unsigned=False
    )
    model = build_semantic_model(module, grammar)

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    rescale = _node_by_label(graph, "RESCALE")
    attrs = {attr.key: attr.value for attr in rescale.attrs}
    assert attrs["arg5 input"] == "uint32[4294967295]"
    assert attrs["arg8 input_zp"] == "uint32[4294967295]"
    assert attrs["arg9 output_zp"] == "int32[-1]"


def test_format_type_formats_array_and_runtime_array_types():
    instruction = _type_int(0, 1, 8)
    uint8_type = ScalarType(
        1, "OpTypeInt", instruction, "int", width=8, signed=False
    )
    length = Constant(
        2,
        1,
        "OpConstant",
        4,
        _constant(1, 1, 2, 4),
        type=uint8_type,
        resolved_value=4,
    )
    shape = Constant(
        3,
        4,
        "OpConstantComposite",
        (2,),
        _constant_composite(2, 4, 3, (2,)),
        resolved_value=(4,),
    )
    tensor_type = TensorArmType(
        5,
        "OpTypeTensorARM",
        _type_tensor(3, 5, 1, 2, 3),
        element_type_id=1,
        rank_id=2,
        shape_id=3,
        element_type=uint8_type,
        rank=length,
        shape=shape,
    )
    array_type = ArrayType(
        6,
        "OpTypeArray",
        _type_array(4, 6, 5, 2),
        element_type_id=5,
        length_id=2,
        element_type=tensor_type,
        length=length,
    )
    runtime_array_type = RuntimeArrayType(
        7,
        "OpTypeRuntimeArray",
        _type_runtime_array(5, 7, 5),
        element_type_id=5,
        element_type=tensor_type,
    )

    assert format_type(array_type, None) == "array<4 x tensor>"
    assert format_type(runtime_array_type, None) == "runtime_array<tensor>"


def test_tensor_shape_metadata_dereferences_constant_composite_dimensions():
    grammar = load_core_grammar()
    module = _module_with_composite_shape_tensor_graph()
    model = build_semantic_model(module, grammar)

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    input_node = _node_by_label(graph, "Input 0")
    output_node = _node_by_label(graph, "Output 0")

    assert graph.groupNodeAttributes[""]["spv graph type id"] == "%10"
    assert graph.groupNodeAttributes[""]["graph type"] == (
        "(uint8[4]) -> (uint8[4])"
    )
    assert "%" not in graph.groupNodeAttributes[""]["graph type"]
    assert _attr_value(input_node, "spv type id") == "%24"
    assert _attr_value(input_node, "type") == "uint8[4]"
    assert _attr_value(input_node, "dtype") == "uint8"
    assert _attr_value(input_node, "shape") == "[4]"
    assert "%" not in _attr_value(input_node, "type")
    assert "%" not in _attr_value(input_node, "dtype")
    assert "%" not in _attr_value(input_node, "shape")
    assert _metadata_attrs(input_node.outputsMetadata) == {
        "0": {"tensor_shape": "uint8[4]"}
    }
    output_attrs = {attr.key: attr.value for attr in output_node.attrs}
    assert "value" not in output_attrs
    assert output_attrs["spv value id"] == "%30"
    assert _metadata_attrs(output_node.inputsMetadata) == {
        "0": {"tensor_shape": "uint8[4]"}
    }


def test_tensor_shape_dimensions_are_not_reinterpreted_as_constant_ids():
    grammar = load_core_grammar()
    module = _module_with_shape_dimensions_matching_constant_ids()
    model = build_semantic_model(module, grammar)

    graph = build_spirv_graphs_from_semantic_model(model, module=module)[0]

    input_node = _node_by_label(graph, "Input 0")
    assert graph.groupNodeAttributes[""]["graph type"] == (
        "(uint8[20, 21]) -> (uint8[20, 21])"
    )
    assert _attr_value(input_node, "type") == "uint8[20, 21]"
    assert _attr_value(input_node, "shape") == "[20, 21]"


def gb_edge(
    source_node_id: str,
    target_input_id: str,
    *,
    source_output_id: str = "0",
):
    from model_explorer import graph_builder as gb

    return gb.IncomingEdge(
        sourceNodeId=source_node_id,
        sourceNodeOutputId=source_output_id,
        targetNodeInputId=target_input_id,
    )


def test_multiple_arm_graphs_create_multiple_model_explorer_graphs():
    fixture = Path(__file__).with_name("fixtures") / "multi-arm-graphs.spvasm"
    assembly = assemble_spirv_text(fixture.read_text(encoding="utf-8"))
    assert assembly.ok, assembly.diagnostics

    graphs = build_spirv_graphs_from_bytes(assembly.data)

    assert [graph.id for graph in graphs] == [
        "graph_11",
        "graph_12",
    ]
    assert [
        graph.groupNodeAttributes[""]["entry_points"] for graph in graphs
    ] == ["first, first_alt", "second, second_alt"]

    first_input, first_output = graphs[0].nodes
    first_input_attrs = _attrs(first_input)
    assert {
        key: first_input_attrs[key]
        for key in (
            "logical input idx",
            "descriptor (first)",
            "descriptor (first_alt)",
            "type",
            "dtype",
            "shape",
            "spv id",
            "spv type id",
        )
    } == {
        "logical input idx": "0",
        "descriptor (first)": "set 0, binding 0",
        "descriptor (first_alt)": "set 7, binding 19",
        "type": "uint8[1]",
        "dtype": "uint8",
        "shape": "[1]",
        "spv id": "%30",
        "spv type id": "%24",
    }
    first_output_attrs = _attrs(first_output)
    assert {
        key: first_output_attrs[key]
        for key in (
            "logical output idx",
            "descriptor (first)",
            "descriptor (first_alt)",
            "spv value id",
        )
    } == {
        "logical output idx": "0",
        "descriptor (first)": "set 0, binding 1",
        "descriptor (first_alt)": "set 7, binding 20",
        "spv value id": "%30",
    }

    second_input, second_output = graphs[1].nodes
    second_input_attrs = _attrs(second_input)
    assert {
        key: second_input_attrs[key]
        for key in (
            "logical input idx",
            "descriptor (second)",
            "descriptor (second_alt)",
            "type",
            "dtype",
            "shape",
            "spv id",
            "spv type id",
        )
    } == {
        "logical input idx": "0",
        "descriptor (second)": "set 1, binding 0",
        "descriptor (second_alt)": "set 8, binding 21",
        "type": "uint8[1]",
        "dtype": "uint8",
        "shape": "[1]",
        "spv id": "%31",
        "spv type id": "%24",
    }
    second_output_attrs = _attrs(second_output)
    assert {
        key: second_output_attrs[key]
        for key in (
            "logical output idx",
            "descriptor (second)",
            "descriptor (second_alt)",
            "spv value id",
        )
    } == {
        "logical output idx": "0",
        "descriptor (second)": "set 1, binding 1",
        "descriptor (second_alt)": "set 8, binding 22",
        "spv value id": "%31",
    }
