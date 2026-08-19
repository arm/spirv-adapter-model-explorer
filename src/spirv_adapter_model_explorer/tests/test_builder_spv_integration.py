# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from spirv_adapter_model_explorer.builder import (
    build_spirv_graph_from_bytes,
    build_spirv_graphs_from_bytes,
)

from .conftest import fixture_text
from .test_spirv_parser import assemble_valid_spirv

MULTI_INPUT_OUTPUT_GRAPH_ASM = fixture_text("data-graph-tosa-two-adds.spvasm")


TENSOR_ARRAY_INTERFACE_GRAPH_ASM = fixture_text(
    "tensor-array-interface.spvasm"
)


TENSOR_ARRAY_SPARSE_INTERFACE_GRAPH_ASM = fixture_text(
    "tensor-array-sparse-interface.spvasm"
)


CONV2D_GRAPH_ASM = fixture_text("conv2d.spvasm")


CONV2D_GRAPH_CONSTANTS_ASM = fixture_text("conv2d-graph-constants.spvasm")


RFFT_STRUCT_OUTPUT_GRAPH_ASM = fixture_text("rfft-struct-output.spvasm")


CONCAT_GRAPH_ASM = fixture_text("concat.spvasm")


SIGNEDNESS_GRAPH_ASM = fixture_text("signedness.spvasm")


def _large_constant_operands_graph_asm(element_count: int = 1024) -> str:
    explicit_scalar_constants = "\n".join(
        f"%bias_{index} = OpConstant %uint "
        f"{0xFFFFFFFF if index == 1 else index}"
        for index in range(1, element_count + 1)
    )
    explicit_constituents = " ".join(
        f"%bias_{index}" for index in range(1, element_count + 1)
    )
    return f"""
; SPIR-V
; Version: 1.0
; Bound: 2000
; Schema: 0
OpCapability Shader
OpCapability TensorsARM
OpCapability GraphARM
OpCapability Int8
OpCapability ReplicatedCompositesEXT
OpExtension "SPV_ARM_tensors"
OpExtension "SPV_ARM_graph"
OpExtension "SPV_EXT_replicated_composites"
%tosa = OpExtInstImport "TOSA.001000.1"
OpMemoryModel Logical GLSL450
OpName %add_explicit "add_explicit_bias"
OpName %add_zero "add_zero_bias"
OpName %add_replicate "add_replicated_bias"

%uint = OpTypeInt 32 0
%uint8 = OpTypeInt 8 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_count = OpConstant %uint {element_count}
{explicit_scalar_constants}
%uint_array_l1 = OpTypeArray %uint %uint_1
%shape = OpConstantComposite %uint_array_l1 %uint_count
%uint8_tensor = OpTypeTensorARM %uint8 %uint_1 %shape
%uint_tensor = OpTypeTensorARM %uint %uint_1 %shape
%explicit_bias = OpConstantComposite %uint_tensor {explicit_constituents}
%zero_bias = OpConstantNull %uint_tensor
%replicated_bias = OpConstantCompositeReplicateEXT %uint_tensor %uint_1
%ptr_tensor = OpTypePointer UniformConstant %uint8_tensor
%graph_type = OpTypeGraphARM 1 %uint8_tensor %uint8_tensor

%input_ptr = OpVariable %ptr_tensor UniformConstant
%output_ptr = OpVariable %ptr_tensor UniformConstant

OpGraphEntryPointARM %graph "main" %input_ptr %output_ptr
%graph = OpGraphARM %graph_type
%input = OpGraphInputARM %uint8_tensor %uint_0
%add_explicit = OpExtInst %uint8_tensor %tosa ADD %input %explicit_bias
%add_zero = OpExtInst %uint8_tensor %tosa ADD %add_explicit %zero_bias
%add_replicate = OpExtInst %uint8_tensor %tosa ADD %add_zero %replicated_bias
OpGraphSetOutputARM %add_replicate %uint_0
OpGraphEndARM
"""


def test_spv_graph_binary_shows_current_large_constant_operand_display(
    tmp_path,
):
    data = assemble_valid_spirv(_large_constant_operands_graph_asm())
    spv_path = tmp_path / "large_constant_operands.spv"
    spv_path.write_bytes(data)

    graph = build_spirv_graphs_from_bytes(data)[0]
    add_explicit, add_zero, add_replicate = [
        node for node in graph.nodes if node.label == "ADD"
    ]

    explicit_attrs = _attrs(add_explicit)
    zero_attrs = _attrs(add_zero)
    replicate_attrs = _attrs(add_replicate)
    rendered_attrs = "\n".join(
        (
            f"explicit bias: {explicit_attrs['arg1 input2']}",
            f"zero bias: {zero_attrs['arg1 input2']}",
            f"replicated bias: {replicate_attrs['arg1 input2']}",
        )
    )
    (tmp_path / "large_constant_attrs.txt").write_text(
        rendered_attrs, encoding="utf-8"
    )

    explicit_preview = ", ".join(str(value) for value in (-1, *range(2, 33)))
    replicate_preview = ", ".join("1" for _index in range(32))
    assert (
        explicit_attrs["arg1 input2"]
        == f"int32[{explicit_preview}, ...] len=1024"
    ), f"SPV written to {spv_path}\n{rendered_attrs}"
    assert zero_attrs["arg1 input2"] == "int32[<all zeroes>] len=1024", (
        f"SPV written to {spv_path}\n{rendered_attrs}"
    )
    assert (
        replicate_attrs["arg1 input2"]
        == f"int32[{replicate_preview}, ...] len=1024"
    ), f"SPV written to {spv_path}\n{rendered_attrs}"


def test_spv_graph_binary_builds_semantic_graph_with_multiple_inputs_outputs():
    data = assemble_valid_spirv(MULTI_INPUT_OUTPUT_GRAPH_ASM)

    graphs = build_spirv_graphs_from_bytes(data)

    assert len(graphs) == 1
    graph = graphs[0]
    assert graph.id.startswith("graph_")
    assert [node.label for node in graph.nodes] == [
        "Input 0",
        "Input 1",
        "ADD",
        "ADD",
        "Output 0",
        "Output 1",
    ]
    assert [node.id for node in graph.nodes] == [
        "input_0",
        "input_1",
        "op_4",
        "op_5",
        "output_0",
        "output_1",
    ]


def test_spv_graph_binary_builds_edges_for_operation_chains():
    data = assemble_valid_spirv(MULTI_INPUT_OUTPUT_GRAPH_ASM)

    graph = build_spirv_graphs_from_bytes(data)[0]
    input0, input1, add0, add1, output0, output1 = graph.nodes

    assert [
        (edge.sourceNodeId, edge.targetNodeInputId)
        for edge in add0.incomingEdges
    ] == [
        (input0.id, "0"),
        (input1.id, "1"),
    ]
    assert [
        (edge.sourceNodeId, edge.targetNodeInputId)
        for edge in add1.incomingEdges
    ] == [
        (add0.id, "0"),
        (input1.id, "1"),
    ]
    assert [
        (edge.sourceNodeId, edge.targetNodeInputId)
        for edge in output0.incomingEdges
    ] == [
        (add0.id, "0"),
    ]
    assert [
        (edge.sourceNodeId, edge.targetNodeInputId)
        for edge in output1.incomingEdges
    ] == [
        (add1.id, "0"),
    ]


def test_spv_graph_binary_applies_namespace_and_node_id_prefix():
    data = assemble_valid_spirv(MULTI_INPUT_OUTPUT_GRAPH_ASM)

    graph = build_spirv_graph_from_bytes(
        data,
        entry_point="main",
        entry_point_kind="graph",
        namespace="0_graph_preprocess",
        node_id_prefix="0__",
    )

    assert all(node.namespace == "0_graph_preprocess" for node in graph.nodes)
    assert all(node.id.startswith("0__") for node in graph.nodes)
    assert all(
        edge.sourceNodeId.startswith("0__")
        for node in graph.nodes
        for edge in node.incomingEdges
    )
    assert "0_graph_preprocess" in graph.groupNodeAttributes
    assert "" not in graph.groupNodeAttributes
    assert (
        graph.groupNodeAttributes["0_graph_preprocess"]["Validation"]
        == "Passed"
    )


def test_spv_graph_binary_displays_tensor_array_interface_elements(tmp_path):
    data = assemble_valid_spirv(TENSOR_ARRAY_INTERFACE_GRAPH_ASM)
    spv_path = tmp_path / "tensor_array_interface.spv"
    spv_path.write_bytes(data)

    graph = build_spirv_graphs_from_bytes(data)[0]
    input0, input1, add0, add1, add2, output0, output1 = graph.nodes

    assert graph.groupNodeAttributes[""]["graph type"] == (
        "(array<2 x uint8[1]>) -> (array<2 x uint8[1]>)"
    )
    assert [node.label for node in graph.nodes] == [
        "Input 0[0]",
        "Input 0[1]",
        "ADD",
        "ADD",
        "ADD",
        "Output 0[0]",
        "Output 0[1]",
    ], f"SPV written to {spv_path}"
    assert [node.id for node in graph.nodes] == [
        "input_0_0",
        "input_0_1",
        "op_16",
        "op_17",
        "op_18",
        "output_0_0",
        "output_0_1",
    ], f"SPV written to {spv_path}"
    assert graph.groupNodeAttributes[""]["debug info"] == "main graph debug"
    assert _attrs(input0)["element index"] == "0"
    assert _attrs(input0)["descriptor (main)"] == "set 0, binding 0"
    assert _attrs(input0)["debug info"] == "input element zero debug"
    assert _attrs(input1)["element index"] == "1"
    assert _attrs(input1)["descriptor (main)"] == "set 0, binding 0"
    assert _attrs(input1)["debug info"] == "input element one debug"
    assert _attrs(add0)["debug info"] == "add0 operation debug"
    assert _attrs(add1)["debug info"] == "add1 operation debug"
    assert _attrs(add2)["debug info"] == "add2 operation debug"
    assert _attrs(output0)["element index"] == "0"
    assert _attrs(output0)["descriptor (main)"] == "set 0, binding 1"
    assert _attrs(output0)["debug info"] == "input element zero debug"
    assert _attrs(output1)["element index"] == "1"
    assert _attrs(output1)["descriptor (main)"] == "set 0, binding 1"
    assert _attrs(output1)["debug info"] == "add2 output debug"
    assert _metadata_attrs(input0.outputsMetadata) == {
        "0": {"tensor_shape": "uint8[1]"}
    }
    assert _metadata_attrs(input1.outputsMetadata) == {
        "0": {"tensor_shape": "uint8[1]"}
    }
    assert _metadata_attrs(add0.inputsMetadata) == {
        "0": {
            "__tensor_tag": "input1",
            "debug info": "input element zero debug",
            "tensor_shape": "uint8[1]",
        },
        "1": {
            "__tensor_tag": "input2",
            "debug info": "input element one debug",
            "tensor_shape": "uint8[1]",
        },
    }
    assert _metadata_attrs(add1.inputsMetadata) == {
        "0": {
            "__tensor_tag": "input1",
            "debug info": "add0 output debug",
            "tensor_shape": "uint8[1]",
        },
        "1": {
            "__tensor_tag": "input2",
            "debug info": "input element one debug",
            "tensor_shape": "uint8[1]",
        },
    }
    assert _metadata_attrs(add2.inputsMetadata) == {
        "0": {
            "__tensor_tag": "input1",
            "debug info": "add1 output debug",
            "tensor_shape": "uint8[1]",
        },
        "1": {
            "__tensor_tag": "input2",
            "debug info": "input element zero debug",
            "tensor_shape": "uint8[1]",
        },
    }
    assert _metadata_attrs(add0.outputsMetadata) == {
        "0": {"debug info": "add0 output debug", "tensor_shape": "uint8[1]"}
    }
    assert _metadata_attrs(add1.outputsMetadata) == {
        "0": {"debug info": "add1 output debug", "tensor_shape": "uint8[1]"}
    }
    assert _metadata_attrs(add2.outputsMetadata) == {
        "0": {"debug info": "add2 output debug", "tensor_shape": "uint8[1]"}
    }
    assert _metadata_attrs(output0.inputsMetadata) == {
        "0": {"tensor_shape": "uint8[1]"}
    }
    assert _metadata_attrs(output1.inputsMetadata) == {
        "0": {"tensor_shape": "uint8[1]"}
    }
    assert [
        (edge.sourceNodeId, edge.targetNodeInputId)
        for edge in add0.incomingEdges
    ] == [
        (input0.id, "0"),
        (input1.id, "1"),
    ]
    assert [
        (edge.sourceNodeId, edge.targetNodeInputId)
        for edge in add1.incomingEdges
    ] == [
        (add0.id, "0"),
        (input1.id, "1"),
    ]
    assert [
        (edge.sourceNodeId, edge.targetNodeInputId)
        for edge in add2.incomingEdges
    ] == [
        (add1.id, "0"),
        (input0.id, "1"),
    ]
    assert [
        (edge.sourceNodeId, edge.sourceNodeOutputId, edge.targetNodeInputId)
        for edge in output0.incomingEdges
    ] == [(input0.id, "0", "0")]
    assert [
        (edge.sourceNodeId, edge.sourceNodeOutputId, edge.targetNodeInputId)
        for edge in output1.incomingEdges
    ] == [(add2.id, "0", "0")]


def test_spv_graph_binary_shows_only_referenced_tensor_array_elements(
    tmp_path,
):
    data = assemble_valid_spirv(TENSOR_ARRAY_SPARSE_INTERFACE_GRAPH_ASM)
    spv_path = tmp_path / "tensor_array_sparse_interface.spv"
    spv_path.write_bytes(data)

    graph = build_spirv_graphs_from_bytes(data)[0]
    input2, output3 = graph.nodes

    assert graph.groupNodeAttributes[""]["graph type"] == (
        "(array<4 x uint8[1]>) -> (array<4 x uint8[1]>)"
    )
    assert [node.label for node in graph.nodes] == [
        "Input 0[2]",
        "Output 0[3]",
    ], f"SPV written to {spv_path}"
    assert [node.id for node in graph.nodes] == [
        "input_0_2",
        "output_0_3",
    ], f"SPV written to {spv_path}"
    assert _attrs(input2)["element index"] == "2"
    assert _attrs(input2)["descriptor (main)"] == "set 0, binding 0"
    assert _attrs(output3)["element index"] == "3"
    assert _attrs(output3)["descriptor (main)"] == "set 0, binding 1"
    assert _metadata_attrs(input2.outputsMetadata) == {
        "0": {"tensor_shape": "uint8[1]"}
    }
    assert _metadata_attrs(output3.inputsMetadata) == {
        "0": {"tensor_shape": "uint8[1]"}
    }
    assert [
        (edge.sourceNodeId, edge.sourceNodeOutputId, edge.targetNodeInputId)
        for edge in output3.incomingEdges
    ] == [(input2.id, "0", "0")]


def test_spv_graph_binary_uses_tosa_operand_names_from_grammar():
    data = assemble_valid_spirv(MULTI_INPUT_OUTPUT_GRAPH_ASM)

    graph = build_spirv_graphs_from_bytes(data)[0]
    input0 = graph.nodes[0]
    add0 = graph.nodes[2]

    input_attrs = {attr.key: attr.value for attr in input0.attrs}
    assert input_attrs["spv type id"].startswith("%")
    assert input_attrs["type"] == "uint8[1]"
    assert input_attrs["dtype"] == "uint8"
    assert input_attrs["shape"] == "[1]"
    assert "%" not in input_attrs["type"]
    assert "%" not in input_attrs["dtype"]
    assert "%" not in input_attrs["shape"]

    attrs = {attr.key: attr.value for attr in add0.attrs}
    assert attrs["spv result type id"].startswith("%")
    assert attrs["result type"] == "uint8[1]"
    assert attrs["spec"] == (
        "[TOSA spec: ADD]("
        "https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html#_add)"
    )
    assert "%" not in attrs["result type"]
    assert "Ext inst set" not in attrs
    assert "Ext inst opcode" not in attrs
    assert attrs["spv ext set"] == "TOSA.001000.1"
    assert attrs["spv ext opcode"] == "14"
    assert "arg0 input1" not in attrs
    assert "arg1 input2" not in attrs
    assert "SPIR-V / operand 0 ID" not in attrs
    assert "SPIR-V / operand 1 ID" not in attrs

    output0 = graph.nodes[4]
    output_attrs = {attr.key: attr.value for attr in output0.attrs}
    assert output_attrs["value"] == "add0"
    assert output_attrs["spv value id"].startswith("%")


def test_spv_graph_binary_includes_full_and_per_node_disassembly():
    data = assemble_valid_spirv(MULTI_INPUT_OUTPUT_GRAPH_ASM)

    graph = build_spirv_graphs_from_bytes(data)[0]
    input0, _input1, add0, _add1, output0, _output1 = graph.nodes
    graph_attrs = graph.groupNodeAttributes[""]
    input_attrs = _attrs(input0)
    add_attrs = _attrs(add0)
    output_attrs = _attrs(output0)

    assert "spv full asm" in graph_attrs
    assert "%add0 = OpExtInst" in graph_attrs["spv full asm"]
    assert "; 0x" in graph_attrs["spv full asm"]

    assert input_attrs["spv byte offset"].startswith("0x")
    assert "OpGraphInputARM" in input_attrs["spv asm"]
    assert (
        input_attrs["spv asm"]
        .rstrip()
        .endswith(f"; {input_attrs['spv byte offset']}")
    )
    assert add_attrs["spv byte offset"].startswith("0x")
    assert "%add0 = OpExtInst" in add_attrs["spv asm"]
    assert (
        add_attrs["spv asm"]
        .rstrip()
        .endswith(f"; {add_attrs['spv byte offset']}")
    )
    assert output_attrs["spv byte offset"].startswith("0x")
    assert "OpGraphSetOutputARM" in output_attrs["spv asm"]
    assert (
        output_attrs["spv asm"]
        .rstrip()
        .endswith(f"; {output_attrs['spv byte offset']}")
    )
    assert int(add_attrs["spv asm line"]) > int(input_attrs["spv asm line"])


def test_spv_graph_binary_formats_tosa_signless_and_rescale_unsigned_operands(
    tmp_path,
):
    # End-to-end coverage for the TOSA signedness display rules using assembled
    # SPIR-V, rather than hand-built semantic objects. TOSA signless integers are
    # encoded as OpTypeInt N 0, so the RESIZE constants below use the raw word
    # 0xffffffff and should display as signed -1. RESCALE then exercises the
    # only TOSA unsigned exception through its input_unsigned/output_unsigned
    # flags.
    data = assemble_valid_spirv(SIGNEDNESS_GRAPH_ASM)
    spv_path = tmp_path / "tosa_signedness.spv"
    spv_path.write_bytes(data)

    graph = build_spirv_graphs_from_bytes(data)[0]
    resize = next(node for node in graph.nodes if node.label == "RESIZE")
    rescale_unsigned, rescale_signed, rescale_mixed = [
        node for node in graph.nodes if node.label == "RESCALE"
    ]

    resize_attrs = _attrs(resize)
    assert resize_attrs["arg2 scale"] == "int32[-1, -1]"
    assert resize_attrs["arg3 offset"] == "int32[-1, -1]"
    assert resize_attrs["arg4 border"] == "int32[-1, -1]"

    unsigned_attrs = _attrs(rescale_unsigned)
    assert unsigned_attrs["arg0 scale32"] == "true"
    assert unsigned_attrs["arg2 per_channel"] == "true"
    assert unsigned_attrs["arg3 input_unsigned"] == "true"
    assert unsigned_attrs["arg4 output_unsigned"] == "true"
    assert unsigned_attrs["arg5 input"] == "uint32[4294967295]"
    assert unsigned_attrs["arg8 input_zp"] == "uint32[4294967295]"
    assert unsigned_attrs["arg9 output_zp"] == "uint32[4294967295]"
    assert unsigned_attrs["arg6 multiplier"] == "int32[-1]"
    assert unsigned_attrs["arg7 shift"] == "int32[-1]"

    signed_attrs = _attrs(rescale_signed)
    assert signed_attrs["arg3 input_unsigned"] == "false"
    assert signed_attrs["arg4 output_unsigned"] == "false"
    assert signed_attrs["arg5 input"] == "int32[-1]"
    assert signed_attrs["arg8 input_zp"] == "int32[-1]"
    assert signed_attrs["arg9 output_zp"] == "int32[-1]"

    mixed_attrs = _attrs(rescale_mixed)
    assert mixed_attrs["arg5 input"] == "uint32[4294967295]"
    assert mixed_attrs["arg8 input_zp"] == "uint32[4294967295]"
    assert mixed_attrs["arg9 output_zp"] == "int32[-1]"


def test_graph_constant_fixture_keeps_attributes_when_nodes_are_hidden():
    data = assemble_valid_spirv(CONV2D_GRAPH_CONSTANTS_ASM)

    graph = build_spirv_graphs_from_bytes(data)[0]
    conv = next(node for node in graph.nodes if node.label == "CONV2D")
    output1 = next(node for node in graph.nodes if node.label == "Output 1")

    assert not any(
        node.label.startswith("<graph constant") for node in graph.nodes
    )
    assert _attrs(conv)["arg6 weight"] == "<graph constant 0> uint8[1]"
    assert _attrs(conv)["arg7 bias"] == "<graph constant 1> uint8[1]"
    assert _attrs(output1)["value"] == "<graph constant 0> uint8[1]"
    assert output1.incomingEdges == []


def test_graph_constant_fixture_shows_connected_nodes_without_attributes():
    data = assemble_valid_spirv(CONV2D_GRAPH_CONSTANTS_ASM)

    graph = build_spirv_graphs_from_bytes(data, show_graph_constants=True)[0]
    nodes_by_id = {node.id: node for node in graph.nodes}
    constants = [
        node
        for node in graph.nodes
        if node.label.startswith("<graph constant")
    ]
    conv = next(node for node in graph.nodes if node.label == "CONV2D")
    output1 = next(node for node in graph.nodes if node.label == "Output 1")

    assert [node.label for node in constants] == [
        "<graph constant 0>",
        "<graph constant 1>",
    ]
    assert "arg6 weight" not in _attrs(conv)
    assert "arg7 bias" not in _attrs(conv)
    assert "value" not in _attrs(output1)
    assert [
        (nodes_by_id[edge.sourceNodeId].label, edge.targetNodeInputId)
        for edge in conv.incomingEdges
    ] == [
        ("Input 0", "0"),
        ("Input 1", "1"),
        ("Input 2", "2"),
        ("Input 3", "3"),
        ("Input 4", "4"),
        ("Input 5", "5"),
        ("<graph constant 0>", "6"),
        ("<graph constant 1>", "7"),
        ("Input 6", "8"),
        ("Input 7", "9"),
    ]
    assert [
        (nodes_by_id[edge.sourceNodeId].label, edge.targetNodeInputId)
        for edge in output1.incomingEdges
    ] == [("<graph constant 0>", "0")]
    assert _metadata_attrs(conv.inputsMetadata)["6"] == {
        "__tensor_tag": "weight",
        "tensor_shape": "uint8[1]",
    }
    assert _metadata_attrs(conv.inputsMetadata)["7"] == {
        "__tensor_tag": "bias",
        "tensor_shape": "uint8[1]",
    }
    assert _metadata_attrs(constants[0].outputsMetadata) == {
        "0": {"tensor_shape": "uint8[1]"}
    }


def test_tosa_operand_names_populate_input_tensor_tags():
    data = assemble_valid_spirv(CONV2D_GRAPH_ASM)

    graph = build_spirv_graphs_from_bytes(data)[0]
    conv = next(node for node in graph.nodes if node.label == "CONV2D")

    assert _metadata_attrs(conv.inputsMetadata) == {
        "0": {"__tensor_tag": "pad", "tensor_shape": "uint8[1]"},
        "1": {"__tensor_tag": "stride", "tensor_shape": "uint8[1]"},
        "2": {"__tensor_tag": "dilation", "tensor_shape": "uint8[1]"},
        "3": {"__tensor_tag": "acc_type", "tensor_shape": "uint8[1]"},
        "4": {"__tensor_tag": "local_bound", "tensor_shape": "uint8[1]"},
        "5": {"__tensor_tag": "input", "tensor_shape": "uint8[1]"},
        "6": {"__tensor_tag": "weight", "tensor_shape": "uint8[1]"},
        "7": {"__tensor_tag": "bias", "tensor_shape": "uint8[1]"},
        "8": {"__tensor_tag": "input_zp", "tensor_shape": "uint8[1]"},
        "9": {"__tensor_tag": "weight_zp", "tensor_shape": "uint8[1]"},
    }


def test_tosa_struct_outputs_expand_and_composite_extracts_select_ports():
    data = assemble_valid_spirv(RFFT_STRUCT_OUTPUT_GRAPH_ASM)

    graph = build_spirv_graphs_from_bytes(data)[0]

    assert [node.label for node in graph.nodes] == [
        "Input 0",
        "Input 1",
        "RFFT2D",
        "Output 0",
        "Output 1",
    ]
    rfft = next(node for node in graph.nodes if node.label == "RFFT2D")
    output0 = next(node for node in graph.nodes if node.label == "Output 0")
    output1 = next(node for node in graph.nodes if node.label == "Output 1")
    rfft_attrs = _attrs(rfft)
    assert "arg0 local_bound" not in rfft_attrs
    assert "arg1 input_real" not in rfft_attrs
    assert _metadata_attrs(rfft.outputsMetadata) == {
        "0": {"__tensor_tag": "output_real", "tensor_shape": "uint8[1]"},
        "1": {"__tensor_tag": "output_imag", "tensor_shape": "uint8[1]"},
    }
    assert [
        (edge.sourceNodeId, edge.sourceNodeOutputId, edge.targetNodeInputId)
        for edge in output0.incomingEdges
    ] == [(rfft.id, "0", "0")]
    assert [
        (edge.sourceNodeId, edge.sourceNodeOutputId, edge.targetNodeInputId)
        for edge in output1.incomingEdges
    ] == [(rfft.id, "1", "0")]


def test_tosa_repeated_operand_names_populate_attrs_and_tensor_tags():
    data = assemble_valid_spirv(CONCAT_GRAPH_ASM)

    graph = build_spirv_graphs_from_bytes(data)[0]
    concat = next(node for node in graph.nodes if node.label == "CONCAT")

    attrs = _attrs(concat)
    assert attrs["arg0 axis"] == "int32[0]"
    assert "arg1 input1" not in attrs
    assert "arg2 input2" not in attrs
    assert "arg3 input3" not in attrs
    assert [
        (edge.sourceNodeId, edge.targetNodeInputId)
        for edge in concat.incomingEdges
    ] == [
        (graph.nodes[0].id, "1"),
        (graph.nodes[1].id, "2"),
        (graph.nodes[2].id, "3"),
    ]
    assert _metadata_attrs(concat.inputsMetadata) == {
        "0": {"__tensor_tag": "axis"},
        "1": {"__tensor_tag": "input1", "tensor_shape": "uint8[1]"},
        "2": {"__tensor_tag": "input2", "tensor_shape": "uint8[1]"},
        "3": {"__tensor_tag": "input3", "tensor_shape": "uint8[1]"},
    }


def test_spv_graph_binary_populates_tensor_shape_metadata_for_all_ports():
    data = assemble_valid_spirv(MULTI_INPUT_OUTPUT_GRAPH_ASM)

    graph = build_spirv_graphs_from_bytes(data)[0]
    input0, input1, add0, add1, output0, output1 = graph.nodes

    tensor_shape = "uint8[1]"
    assert _attrs(input0)["descriptor (main)"] == "set 0, binding 0"
    assert _attrs(input1)["descriptor (main)"] == "set 0, binding 1"
    assert _attrs(output0)["descriptor (main)"] == "set 0, binding 2"
    assert _attrs(output1)["descriptor (main)"] == "set 0, binding 3"
    assert _metadata_attrs(input0.outputsMetadata) == {
        "0": {"tensor_shape": tensor_shape}
    }
    assert _metadata_attrs(input1.outputsMetadata) == {
        "0": {"tensor_shape": tensor_shape}
    }
    assert _metadata_attrs(add0.inputsMetadata) == {
        "0": {"__tensor_tag": "input1", "tensor_shape": tensor_shape},
        "1": {"__tensor_tag": "input2", "tensor_shape": tensor_shape},
    }
    assert _metadata_attrs(add0.outputsMetadata) == {
        "0": {"tensor_shape": tensor_shape}
    }
    assert _metadata_attrs(add1.inputsMetadata) == {
        "0": {"__tensor_tag": "input1", "tensor_shape": tensor_shape},
        "1": {"__tensor_tag": "input2", "tensor_shape": tensor_shape},
    }
    assert _metadata_attrs(add1.outputsMetadata) == {
        "0": {"tensor_shape": tensor_shape}
    }
    assert _metadata_attrs(output0.inputsMetadata) == {
        "0": {"tensor_shape": tensor_shape}
    }
    assert _metadata_attrs(output1.inputsMetadata) == {
        "0": {"tensor_shape": tensor_shape}
    }


def _attrs(node):
    return {attr.key: attr.value for attr in node.attrs}


def _metadata_attrs(metadata_items):
    return {
        item.id: {attr.key: attr.value for attr in item.attrs}
        for item in metadata_items
    }
