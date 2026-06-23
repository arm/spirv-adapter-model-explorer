# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

from model_explorer import graph_builder as gb

from .assembly import SpirvDisassemblyLine, disassemble_spirv
from .disassembly_metadata import (
    disassembly_attrs,
    graph_group_attrs,
)
from .formatting import (
    definition_name_attrs,
    definition_type,
    format_constant_operand_value,
    format_constant_value,
    format_id,
    format_type,
    tensor_attrs,
    tensor_shape_metadata_value,
)
from .raw_graph import (
    build_raw_instruction_graph,
)
from .spirv.semantic_model import build_semantic_model
from .spirv.spirv_grammar import load_core_grammar
from .spirv.spirv_parser import SpirvModule, parse_spirv
from .spirv.types import (
    Constant,
    Decoration,
    Definition,
    GraphArm,
    GraphArmInput,
    GraphArmOperation,
    GraphArmOutput,
    SpirvModuleModel,
    StructType,
    Type,
)
from .tosa import (
    constant_operand_integer_interpretation,
    operation_input_tags,
    operation_output_tags,
    operation_tosa_spec_attr,
)
from .validation import (
    attach_validation_metadata,
    parsing_error_with_validation_context,
    validate_for_metadata,
)


class _ProducerOutput(NamedTuple):
    node_id: str
    output_id: str = "0"


def build_spirv_graphs_from_bytes(
    data: bytes,
    *,
    namespace: str = "",
    node_id_prefix: str = "",
) -> list[gb.Graph]:
    """Build Model Explorer graphs from SPIR-V bytes."""
    validation_result = validate_for_metadata(data)
    try:
        grammar = load_core_grammar()
        module = parse_spirv(data, grammar)
        model = build_semantic_model(module, grammar)
        disassembly = disassemble_spirv(data, show_byte_offset=True)
        graphs = build_spirv_graphs_from_semantic_model(
            model,
            module=module,
            namespace=namespace,
            node_id_prefix=node_id_prefix,
            disassembly_text=disassembly.text if disassembly.ok else None,
            disassembly_lines_by_byte_offset=disassembly.lines_by_byte_offset,
        )
    except Exception as err:
        raise parsing_error_with_validation_context(
            err, validation_result
        ) from err

    for graph in graphs:
        attach_validation_metadata(
            graph, validation_result, namespace=namespace
        )
    return graphs


def build_spirv_graph_from_bytes(
    data: bytes,
    *,
    entry_point: str,
    namespace: str = "",
    node_id_prefix: str = "",
) -> gb.Graph:
    """Build the Model Explorer graph for a named GraphARM entry point."""
    validation_result = validate_for_metadata(data)
    try:
        grammar = load_core_grammar()
        module = parse_spirv(data, grammar)
        model = build_semantic_model(module, grammar)
        disassembly = disassemble_spirv(data, show_byte_offset=True)
        graph = build_spirv_graph_from_semantic_model(
            model,
            entry_point=entry_point,
            namespace=namespace,
            node_id_prefix=node_id_prefix,
            disassembly_text=disassembly.text if disassembly.ok else None,
            disassembly_lines_by_byte_offset=disassembly.lines_by_byte_offset,
        )
    except Exception as err:
        raise parsing_error_with_validation_context(
            err, validation_result
        ) from err

    attach_validation_metadata(graph, validation_result, namespace=namespace)
    return graph


def build_spirv_graph_from_semantic_model(
    model: SpirvModuleModel,
    *,
    entry_point: str,
    namespace: str = "",
    node_id_prefix: str = "",
    disassembly_text: str | None = None,
    disassembly_lines_by_byte_offset: dict[int, SpirvDisassemblyLine]
    | None = None,
) -> gb.Graph:
    """Build a Model Explorer graph for a named GraphARM entry point."""
    graph = _graph_for_entry_point(model, entry_point)
    return _namespace_and_prefix_graph(
        _build_arm_graph(
            graph,
            model,
            entry_point_names=(entry_point,),
            disassembly_text=disassembly_text,
            disassembly_lines_by_byte_offset=disassembly_lines_by_byte_offset,
        ),
        namespace=namespace,
        node_id_prefix=node_id_prefix,
    )


def build_spirv_graphs_from_semantic_model(
    model: SpirvModuleModel,
    *,
    module: SpirvModule | None = None,
    namespace: str = "",
    node_id_prefix: str = "",
    disassembly_text: str | None = None,
    disassembly_lines_by_byte_offset: dict[int, SpirvDisassemblyLine]
    | None = None,
) -> list[gb.Graph]:
    """Build Model Explorer graphs from a semantic SPIR-V model."""
    if model.graphs:
        graphs = list(model.graphs.values())
        return [
            _namespace_and_prefix_graph(
                _build_arm_graph(
                    arm_graph,
                    model,
                    disassembly_text=disassembly_text,
                    disassembly_lines_by_byte_offset=disassembly_lines_by_byte_offset,
                ),
                namespace=namespace,
                node_id_prefix=node_id_prefix,
            )
            for arm_graph in graphs
        ]

    if module is None:
        module = SpirvModule(
            header=model.header,
            instructions=model.global_instructions,
        )
    return [
        _namespace_and_prefix_graph(
            build_raw_instruction_graph(
                module,
                disassembly_text=disassembly_text,
                disassembly_lines_by_byte_offset=disassembly_lines_by_byte_offset,
            ),
            namespace=namespace,
            node_id_prefix=node_id_prefix,
        )
    ]


def _graph_for_entry_point(
    model: SpirvModuleModel, entry_point_name: str
) -> GraphArm:
    graph_entry_points = [
        entry_point
        for entry_point in model.graph_entry_points
        if entry_point.name == entry_point_name
    ]
    if not graph_entry_points:
        raise ValueError(
            f"No OpGraphEntryPointARM named {entry_point_name!r} was found."
        )
    if len(graph_entry_points) > 1:
        raise ValueError(
            "Multiple OpGraphEntryPointARM instructions named "
            f"{entry_point_name!r} were found."
        )
    graph = graph_entry_points[0].graph
    if graph is None:
        raise ValueError(
            f"OpGraphEntryPointARM {entry_point_name!r} does not reference a "
            "known OpGraphARM."
        )
    return graph


def _namespace_and_prefix_graph(
    graph: gb.Graph, *, namespace: str, node_id_prefix: str
) -> gb.Graph:
    if namespace and graph.groupNodeAttributes:
        graph.groupNodeAttributes = {
            namespace if group == "" else group: attrs
            for group, attrs in graph.groupNodeAttributes.items()
        }

    if not namespace and not node_id_prefix:
        return graph

    id_map = {node.id: f"{node_id_prefix}{node.id}" for node in graph.nodes}
    for node in graph.nodes:
        node.id = id_map[node.id]
        if namespace:
            node.namespace = namespace
    for node in graph.nodes:
        for edge in node.incomingEdges:
            edge.sourceNodeId = id_map.get(
                edge.sourceNodeId, edge.sourceNodeId
            )
    return graph


def _build_arm_graph(
    graph: GraphArm,
    model: SpirvModuleModel,
    *,
    entry_point_names: tuple[str, ...] | None = None,
    disassembly_text: str | None = None,
    disassembly_lines_by_byte_offset: dict[int, SpirvDisassemblyLine]
    | None = None,
) -> gb.Graph:
    producer_outputs: dict[int, _ProducerOutput] = {}
    operations_by_id = {
        operation.id: operation for operation in graph.operations
    }
    nodes: list[gb.GraphNode] = []

    for graph_input in graph.inputs:
        node = _graph_input_node(
            graph,
            graph_input,
            model,
            entry_point_names=entry_point_names,
            disassembly_lines_by_byte_offset=disassembly_lines_by_byte_offset,
        )
        nodes.append(node)
        producer_outputs[graph_input.id] = _ProducerOutput(node.id)

    for operation in graph.operations:
        selector = _composite_extract_selector(operation, operations_by_id)
        if selector is not None:
            source_id, output_id = selector
            source_output = producer_outputs.get(source_id)
            if source_output is not None:
                producer_outputs[operation.id] = _ProducerOutput(
                    source_output.node_id, output_id
                )
                continue

        node = _graph_operation_node(
            graph,
            operation,
            producer_outputs,
            model,
            disassembly_lines_by_byte_offset=disassembly_lines_by_byte_offset,
        )
        nodes.append(node)
        producer_outputs[operation.id] = _ProducerOutput(node.id)

    for output in graph.outputs:
        nodes.append(
            _graph_output_node(
                graph,
                output,
                producer_outputs,
                model,
                entry_point_names=entry_point_names,
                disassembly_lines_by_byte_offset=disassembly_lines_by_byte_offset,
            )
        )

    group_attrs = {
        "graph type": format_type(graph.type, model),
        "entry_points": ", ".join(
            entry_point.name for entry_point in graph.entry_points
        ),
        "spv graph id": format_id(graph.id),
        "spv graph type id": format_id(graph.type_id),
    }
    if graph.debug_info:
        group_attrs["debug info"] = _format_debug_info(graph.debug_info)
    return gb.Graph(
        id=_graph_id(graph),
        nodes=nodes,
        groupNodeAttributes=graph_group_attrs(group_attrs, disassembly_text),
    )


def _graph_id(graph: GraphArm) -> str:
    return f"graph_{graph.id}"


def _graph_input_node(
    graph: GraphArm,
    graph_input: GraphArmInput,
    model: SpirvModuleModel,
    *,
    entry_point_names: tuple[str, ...] | None = None,
    disassembly_lines_by_byte_offset: dict[int, SpirvDisassemblyLine]
    | None = None,
) -> gb.GraphNode:
    label = _format_graph_interface_label(
        "Input", graph_input.input_index, graph_input.element_index
    )
    attrs = [
        gb.KeyValue(
            key="logical input idx",
            value=format_constant_value(graph_input.input_index),
        ),
    ]
    attrs.extend(
        _element_index_attrs(
            graph_input.element_index_id, graph_input.element_index
        )
    )
    attrs.extend(
        _descriptor_attrs(
            _graph_input_interface_descriptors(
                graph, graph_input, model, entry_point_names
            )
        )
    )
    attrs.extend(_debug_info_attrs(graph_input.debug_info))
    attrs.extend(
        [
            gb.KeyValue(
                key="type", value=format_type(graph_input.type, model)
            ),
            *tensor_attrs(graph_input.type),
            gb.KeyValue(key="spv id", value=format_id(graph_input.id)),
            gb.KeyValue(
                key="spv type id", value=format_id(graph_input.type_id)
            ),
            *disassembly_attrs(
                graph_input.instruction, disassembly_lines_by_byte_offset
            ),
        ]
    )
    return gb.GraphNode(
        id=_input_node_id(graph_input),
        label=label,
        attrs=attrs,
        outputsMetadata=_port_metadata((graph_input.type,)),
    )


def _graph_operation_node(
    graph: GraphArm,
    operation: GraphArmOperation,
    producer_outputs: dict[int, _ProducerOutput],
    model: SpirvModuleModel,
    *,
    disassembly_lines_by_byte_offset: dict[int, SpirvDisassemblyLine]
    | None = None,
) -> gb.GraphNode:
    attrs = [
        gb.KeyValue(key="opcode", value=operation.opcode_name),
        gb.KeyValue(
            key="result type", value=format_type(operation.type, model)
        ),
    ]
    tosa_spec_attr = operation_tosa_spec_attr(operation)
    if tosa_spec_attr is not None:
        attrs.append(tosa_spec_attr)
    attrs.extend(_debug_info_attrs(operation.debug_info))
    attrs.extend(_operation_operand_attrs(operation, model, producer_outputs))

    spirv_attrs = [
        gb.KeyValue(key="spv id", value=format_id(operation.id)),
        gb.KeyValue(
            key="spv result type id",
            value=format_id(operation.type_id),
        ),
    ]
    if operation.ext_inst_set_name is not None:
        spirv_attrs.append(
            gb.KeyValue(
                key="spv ext set",
                value=operation.ext_inst_set_name,
            )
        )
    if operation.ext_inst_opcode is not None:
        spirv_attrs.append(
            gb.KeyValue(
                key="spv ext opcode",
                value=str(operation.ext_inst_opcode),
            )
        )
    spirv_attrs.extend(
        disassembly_attrs(
            operation.instruction, disassembly_lines_by_byte_offset
        )
    )
    attrs.extend(spirv_attrs)

    return gb.GraphNode(
        id=_operation_node_id(operation),
        label=operation.opcode_name,
        attrs=attrs,
        incomingEdges=_incoming_edges_for_operation(
            operation, producer_outputs
        ),
        inputsMetadata=_port_metadata(
            (
                definition_type(model.definitions.get(operand_id), model)
                for operand_id in operation.operand_ids
            ),
            tags=operation_input_tags(operation),
            debug_info=_operation_input_debug_info(operation, graph),
        ),
        outputsMetadata=_port_metadata(
            _operation_output_types(operation),
            tags=operation_output_tags(operation),
            debug_info=operation.output_debug_info,
        ),
    )


def _graph_output_node(
    graph: GraphArm,
    output: GraphArmOutput,
    producer_outputs: dict[int, _ProducerOutput],
    model: SpirvModuleModel,
    *,
    entry_point_names: tuple[str, ...] | None = None,
    disassembly_lines_by_byte_offset: dict[int, SpirvDisassemblyLine]
    | None = None,
) -> gb.GraphNode:
    producer_output = producer_outputs.get(output.value_id)
    attrs = [
        gb.KeyValue(
            key="logical output idx",
            value=format_constant_value(output.output_index),
        ),
    ]
    attrs.extend(
        _element_index_attrs(output.element_index_id, output.element_index)
    )
    attrs.extend(
        _descriptor_attrs(
            _graph_output_interface_descriptors(
                graph, output, model, entry_point_names
            )
        )
    )
    attrs.extend(
        _debug_info_attrs(
            _dedupe_debug_info(
                (
                    *_graph_value_debug_info(graph, output.value_id),
                    *output.debug_info,
                )
            )
        )
    )
    attrs.extend(
        [
            *definition_name_attrs(
                "value", output.value_id, output.value, model
            ),
            gb.KeyValue(key="spv value id", value=format_id(output.value_id)),
            *disassembly_attrs(
                output.instruction, disassembly_lines_by_byte_offset
            ),
        ]
    )

    return gb.GraphNode(
        id=_output_node_id(output),
        label=_format_graph_interface_label(
            "Output", output.output_index, output.element_index
        ),
        attrs=attrs,
        incomingEdges=_incoming_edges_for_producer(producer_output),
        inputsMetadata=_port_metadata((definition_type(output.value, model),)),
    )


def _format_graph_interface_label(
    prefix: str,
    index: Constant | None,
    element_index: Constant | None,
) -> str:
    label = f"{prefix} {format_constant_value(index)}"
    if element_index is not None:
        label += f"[{format_constant_value(element_index)}]"
    return label


def _element_index_attrs(
    element_index_id: int | None,
    element_index: Constant | None,
) -> list[gb.KeyValue]:
    if element_index_id is None:
        return []
    return [
        gb.KeyValue(
            key="element index",
            value=format_constant_value(element_index),
        ),
        gb.KeyValue(
            key="spv element index id", value=format_id(element_index_id)
        ),
    ]


def _debug_info_attrs(debug_info: tuple[str, ...]) -> list[gb.KeyValue]:
    if not debug_info:
        return []
    return [
        gb.KeyValue(key="debug info", value=_format_debug_info(debug_info))
    ]


def _format_debug_info(debug_info: tuple[str, ...]) -> str:
    return ", ".join(debug_info)


def _dedupe_debug_info(debug_info: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(debug_info))


def _graph_value_debug_info(graph: GraphArm, value_id: int) -> tuple[str, ...]:
    for graph_input in graph.inputs:
        if graph_input.id == value_id:
            return graph_input.debug_info
    for operation in graph.operations:
        if operation.id != value_id:
            continue
        if operation.output_debug_info:
            return operation.output_debug_info[0]
        return operation.debug_info
    return ()


def _graph_input_interface_descriptors(
    graph: GraphArm,
    graph_input: GraphArmInput,
    model: SpirvModuleModel,
    entry_point_names: tuple[str, ...] | None,
) -> tuple[tuple[str, str], ...]:
    input_index = _constant_int_value(graph_input.input_index)
    if input_index is None:
        return ()
    return _entry_point_interface_descriptors(
        graph, input_index, model, entry_point_names
    )


def _graph_output_interface_descriptors(
    graph: GraphArm,
    graph_output: GraphArmOutput,
    model: SpirvModuleModel,
    entry_point_names: tuple[str, ...] | None,
) -> tuple[tuple[str, str], ...]:
    output_index = _constant_int_value(graph_output.output_index)
    input_count = _graph_input_count(graph)
    if output_index is None or input_count is None:
        return ()
    return _entry_point_interface_descriptors(
        graph, input_count + output_index, model, entry_point_names
    )


def _entry_point_interface_descriptors(
    graph: GraphArm,
    interface_index: int,
    model: SpirvModuleModel,
    entry_point_names: tuple[str, ...] | None,
) -> tuple[tuple[str, str], ...]:
    descriptors: list[tuple[str, str]] = []
    for entry_point in graph.entry_points:
        if (
            entry_point_names is not None
            and entry_point.name not in entry_point_names
        ):
            continue
        if interface_index >= len(entry_point.interface_ids):
            continue
        descriptor = _interface_descriptor(
            entry_point.interface_ids[interface_index], model
        )
        if descriptor is not None:
            descriptors.append((entry_point.name, descriptor))
    return tuple(descriptors)


def _graph_input_count(graph: GraphArm) -> int | None:
    if graph.type is None:
        return None
    return len(graph.type.input_type_ids)


def _constant_int_value(constant: Constant | None) -> int | None:
    if constant is None or not isinstance(constant.resolved_value, int):
        return None
    return constant.resolved_value


def _interface_descriptor(
    interface_id: int, model: SpirvModuleModel
) -> str | None:
    descriptor_set = _decoration_operand(
        model.decorations.by_target.get(interface_id, ()), "DescriptorSet"
    )
    binding = _decoration_operand(
        model.decorations.by_target.get(interface_id, ()), "Binding"
    )
    if descriptor_set is None or binding is None:
        return None
    return f"set {descriptor_set}, binding {binding}"


def _decoration_operand(
    decorations: Iterable[Decoration], decoration_name: str
) -> object | None:
    for decoration in decorations:
        if decoration.decoration == decoration_name and decoration.operands:
            return decoration.operands[0]
    return None


def _descriptor_attrs(
    descriptors: Iterable[tuple[str, str]],
) -> list[gb.KeyValue]:
    return [
        gb.KeyValue(key=f"descriptor ({entry_point_name})", value=descriptor)
        for entry_point_name, descriptor in descriptors
    ]


def _incoming_edges_for_producer(
    producer_output: _ProducerOutput | None,
    *,
    target_input_id: str = "0",
) -> list[gb.IncomingEdge]:
    if producer_output is None:
        return []
    return [
        gb.IncomingEdge(
            sourceNodeId=producer_output.node_id,
            sourceNodeOutputId=producer_output.output_id,
            targetNodeInputId=target_input_id,
        )
    ]


def _incoming_edges_for_operation(
    operation: GraphArmOperation, producer_outputs: dict[int, _ProducerOutput]
) -> list[gb.IncomingEdge]:
    incoming_edges: list[gb.IncomingEdge] = []
    for operand_index, operand_id in enumerate(operation.operand_ids):
        incoming_edges.extend(
            _incoming_edges_for_producer(
                producer_outputs.get(operand_id),
                target_input_id=str(operand_index),
            )
        )
    return incoming_edges


def _operation_input_debug_info(
    operation: GraphArmOperation, graph: GraphArm
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        _graph_value_debug_info(graph, operand_id)
        for operand_id in operation.operand_ids
    )


def _operation_output_types(
    operation: GraphArmOperation,
) -> tuple[Type | None, ...]:
    if isinstance(operation.type, StructType):
        return operation.type.member_types
    return (operation.type,)


def _composite_extract_selector(
    operation: GraphArmOperation,
    operations_by_id: dict[int, GraphArmOperation],
) -> tuple[int, str] | None:
    if operation.instruction.opname != "OpCompositeExtract":
        return None
    if len(operation.instruction.operands) != 4:
        return None
    composite_operand = operation.instruction.operands[2]
    composite_id = composite_operand.ref_id
    member_index = _literal_operand(operation.instruction.operands[3].display)
    if composite_id is None or member_index is None:
        return None

    source_operation = operations_by_id.get(composite_id)
    if source_operation is None:
        return None
    source_type = source_operation.type
    if not isinstance(source_type, StructType):
        return None
    if member_index < 0 or member_index >= len(source_type.member_types):
        return None
    member_type = source_type.member_types[member_index]
    if member_type is not None and operation.type is not member_type:
        return None
    return composite_id, str(member_index)


def _literal_operand(display: str) -> int | None:
    try:
        return int(display, 0)
    except ValueError:
        return None


def _operation_operand_attrs(
    operation: GraphArmOperation,
    model: SpirvModuleModel,
    producer_outputs: dict[int, _ProducerOutput] | None = None,
) -> list[gb.KeyValue]:
    attrs: list[gb.KeyValue] = []
    operand_tags = operation_input_tags(operation)
    for index, operand_id in enumerate(operation.operand_ids):
        key = f"arg{index}"
        if index < len(operand_tags) and operand_tags[index]:
            key = f"arg{index} {operand_tags[index]}"
        operand = (
            operation.operands[index]
            if index < len(operation.operands)
            else None
        )
        if isinstance(operand, Definition):
            graph_constant_attr = _graph_constant_operand_attr(
                key, operand, model
            )
            if graph_constant_attr is not None:
                attrs.append(graph_constant_attr)
            elif (
                producer_outputs is None or operand_id not in producer_outputs
            ):
                attrs.extend(
                    definition_name_attrs(key, operand_id, operand, model)
                )
        elif operand_id in model.constants:
            attrs.append(
                gb.KeyValue(
                    key=key,
                    value=format_constant_operand_value(
                        model.constants[operand_id],
                        model,
                        integer_interpretation=constant_operand_integer_interpretation(
                            operation, index, model
                        ),
                    ),
                )
            )
    return attrs


def _graph_constant_operand_attr(
    key: str, definition: Definition, model: SpirvModuleModel
) -> gb.KeyValue | None:
    if definition.instruction.opname != "OpGraphConstantARM":
        return None
    if len(definition.instruction.operands) < 3:
        return None
    graph_constant_id = definition.instruction.operands[2].display
    value = f"<graph constant {graph_constant_id}>"
    type_model = definition_type(definition, model)
    formatted_type = format_type(type_model, model)
    if formatted_type:
        value += f" {formatted_type}"
    return gb.KeyValue(key=key, value=value)


def _port_metadata(
    type_models: Iterable[Type | None],
    *,
    tags: Iterable[str | None] = (),
    debug_info: Iterable[tuple[str, ...]] = (),
) -> list[gb.MetadataItem]:
    metadata: list[gb.MetadataItem] = []
    tag_by_index = dict(enumerate(tags))
    debug_info_by_index = dict(enumerate(debug_info))
    for index, type_model in enumerate(type_models):
        attrs: list[gb.KeyValue] = []
        tag = tag_by_index.get(index)
        if tag:
            attrs.append(gb.KeyValue(key="__tensor_tag", value=tag))
        port_debug_info = debug_info_by_index.get(index, ())
        if port_debug_info:
            attrs.extend(_debug_info_attrs(port_debug_info))
        tensor_shape = tensor_shape_metadata_value(type_model)
        if tensor_shape is not None:
            attrs.append(gb.KeyValue(key="tensor_shape", value=tensor_shape))
        metadata.append(gb.MetadataItem(id=str(index), attrs=attrs))
    return metadata


def _input_node_id(graph_input: GraphArmInput) -> str:
    return _indexed_node_id(
        "input",
        graph_input.input_index,
        graph_input.input_index_id,
        graph_input.element_index,
        graph_input.element_index_id,
    )


def _operation_node_id(operation: GraphArmOperation) -> str:
    return f"op_{operation.id}"


def _output_node_id(output: GraphArmOutput) -> str:
    return _indexed_node_id(
        "output",
        output.output_index,
        output.output_index_id,
        output.element_index,
        output.element_index_id,
    )


def _indexed_node_id(
    prefix: str,
    index: Constant | None,
    index_id: int | None,
    element_index: Constant | None,
    element_index_id: int | None,
) -> str:
    parts = [prefix, _node_id_index_part(index, index_id)]
    if element_index_id is not None:
        parts.append(_node_id_index_part(element_index, element_index_id))
    return "_".join(parts)


def _node_id_index_part(index: Constant | None, index_id: int | None) -> str:
    value = _constant_int_value(index)
    if value is not None:
        return str(value)
    if index_id is not None:
        return str(index_id)
    return "unknown"


__all__ = [
    "build_spirv_graph_from_bytes",
    "build_spirv_graph_from_semantic_model",
    "build_spirv_graphs_from_bytes",
    "build_spirv_graphs_from_semantic_model",
]
