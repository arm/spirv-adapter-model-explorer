# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

from model_explorer import graph_builder as gb

from .assembly import SpirvDisassemblyLine
from .disassembly_metadata import (
    disassembly_attrs,
    graph_group_attrs,
)
from .formatting import (
    definition_name_attrs,
    format_id,
    format_type,
)
from .spirv.spirv_parser import SpirvInstruction
from .spirv.types import (
    Decoration,
    EntryPoint,
    Function,
    SpirvModuleModel,
    Variable,
)


class _ComputeResource(NamedTuple):
    variable: Variable
    descriptor_set: object
    binding: object
    input_index: int | None
    output_index: int | None


def build_compute_graph(
    model: SpirvModuleModel,
    entry_point: EntryPoint,
    *,
    disassembly_text: str | None = None,
    disassembly_lines_by_byte_offset: dict[int, SpirvDisassemblyLine]
    | None = None,
) -> gb.Graph:
    """Build a compact Model Explorer graph for a compute entry point."""

    resources = _compute_resources(model, entry_point)
    compute_node = _compute_node(
        entry_point,
        resources,
        model,
        disassembly_lines_by_byte_offset=disassembly_lines_by_byte_offset,
    )
    nodes = [
        *(
            _compute_resource_node(
                resource,
                model,
                disassembly_lines_by_byte_offset=(
                    disassembly_lines_by_byte_offset
                ),
            )
            for resource in resources
        ),
        compute_node,
    ]
    return gb.Graph(
        id="compute",
        nodes=nodes,
        groupNodeAttributes=graph_group_attrs(
            _compute_group_attrs(entry_point, resources), disassembly_text
        ),
    )


def _compute_resources(
    model: SpirvModuleModel, entry_point: EntryPoint
) -> tuple[_ComputeResource, ...]:
    referenced_variable_ids = _referenced_global_variable_ids(
        model, entry_point
    )
    descriptor_variables = _module_descriptor_variables(model)
    if referenced_variable_ids:
        descriptor_variables = [
            descriptor_variable
            for descriptor_variable in descriptor_variables
            if descriptor_variable[0].id in referenced_variable_ids
        ]
    descriptor_variables.sort(
        key=lambda item: (*_descriptor_sort_key(item[1], item[2]), item[0].id)
    )

    input_index = 0
    output_index = 0
    resources: list[_ComputeResource] = []
    for variable, descriptor_set, binding in descriptor_variables:
        is_input = _is_compute_input(variable, model)
        is_output = _is_compute_output(variable, model)
        resource = _ComputeResource(
            variable=variable,
            descriptor_set=descriptor_set,
            binding=binding,
            input_index=input_index if is_input else None,
            output_index=output_index if is_output else None,
        )
        resources.append(resource)
        if is_input:
            input_index += 1
        if is_output:
            output_index += 1
    return tuple(resources)


def _module_descriptor_variables(
    model: SpirvModuleModel,
) -> list[tuple[Variable, object, object]]:
    return [
        (
            variable,
            descriptor_set if descriptor_set is not None else 0,
            binding,
        )
        for variable in model.variables.values()
        for descriptor_set, binding in [_descriptor(variable, model)]
        if binding is not None
    ]


def _referenced_global_variable_ids(
    model: SpirvModuleModel, entry_point: EntryPoint
) -> set[int]:
    referenced_ids = {
        interface_id
        for interface_id in entry_point.interface_ids
        if interface_id in model.variables
    }
    for function_id in _reachable_function_ids(model, entry_point.id):
        function = model.functions.get(function_id)
        if function is None:
            continue
        for instruction in _function_body_instructions(function):
            referenced_ids.update(
                operand.ref_id
                for operand in instruction.operands
                if operand.ref_id in model.variables
            )
    return referenced_ids


def _reachable_function_ids(
    model: SpirvModuleModel, entry_point_function_id: int
) -> set[int]:
    pending = [entry_point_function_id]
    reachable: set[int] = set()
    while pending:
        function_id = pending.pop()
        if function_id in reachable:
            continue
        reachable.add(function_id)
        function = model.functions.get(function_id)
        if function is None:
            continue
        for instruction in _function_body_instructions(function):
            if instruction.opname != "OpFunctionCall":
                continue
            pending.extend(
                operand.ref_id
                for operand in instruction.operands
                if operand.ref_id in model.functions
                and operand.ref_id not in reachable
            )
    return reachable


def _function_body_instructions(
    function: Function,
) -> Iterable[SpirvInstruction]:
    return (
        instruction
        for block in function.blocks
        for instruction in block.instructions
    )


def _descriptor(
    variable: Variable, model: SpirvModuleModel
) -> tuple[object | None, object | None]:
    decorations = model.decorations.by_target.get(variable.id, ())
    return (
        _decoration_operand(decorations, "DescriptorSet"),
        _decoration_operand(decorations, "Binding"),
    )


def _descriptor_sort_key(
    descriptor_set: object, binding: object
) -> tuple[int, int]:
    return (_sort_int(descriptor_set), _sort_int(binding))


def _sort_int(value: object) -> int:
    return value if isinstance(value, int) else 0


def _is_compute_input(variable: Variable, model: SpirvModuleModel) -> bool:
    return not _has_decoration(variable, model, "NonReadable")


def _is_compute_output(variable: Variable, model: SpirvModuleModel) -> bool:
    return not _has_decoration(variable, model, "NonWritable")


def _has_decoration(
    variable: Variable, model: SpirvModuleModel, decoration_name: str
) -> bool:
    return any(
        decoration.decoration == decoration_name
        for decoration in model.decorations.by_target.get(variable.id, ())
    )


def _compute_resource_node(
    resource: _ComputeResource,
    model: SpirvModuleModel,
    *,
    disassembly_lines_by_byte_offset: dict[int, SpirvDisassemblyLine]
    | None = None,
) -> gb.GraphNode:
    variable = resource.variable
    attrs = [
        gb.KeyValue(
            key="descriptor",
            value=_format_descriptor(
                resource.descriptor_set, resource.binding
            ),
        ),
        gb.KeyValue(key="storage class", value=variable.storage_class),
        *definition_name_attrs(
            "name", variable.id, model.definitions.get(variable.id), model
        ),
        gb.KeyValue(key="type", value=format_type(variable.type, model)),
        gb.KeyValue(key="spv id", value=format_id(variable.id)),
        gb.KeyValue(key="spv type id", value=format_id(variable.type_id)),
    ]
    attrs.extend(
        disassembly_attrs(
            variable.instruction, disassembly_lines_by_byte_offset
        )
    )

    incoming_edges = []
    if resource.output_index is not None:
        incoming_edges.append(
            gb.IncomingEdge(
                sourceNodeId=_COMPUTE_NODE_ID,
                sourceNodeOutputId=str(resource.output_index),
                targetNodeInputId="0",
            )
        )

    return gb.GraphNode(
        id=_compute_resource_node_id(resource),
        label=_compute_resource_label(resource, model),
        attrs=attrs,
        incomingEdges=incoming_edges,
        outputsMetadata=[gb.MetadataItem(id="0", attrs=[])],
    )


_COMPUTE_NODE_ID = "compute"


def _compute_node(
    entry_point: EntryPoint,
    resources: Iterable[_ComputeResource],
    model: SpirvModuleModel,
    *,
    disassembly_lines_by_byte_offset: dict[int, SpirvDisassemblyLine]
    | None = None,
) -> gb.GraphNode:
    input_resources = [
        resource for resource in resources if resource.input_index is not None
    ]
    output_resources = [
        resource for resource in resources if resource.output_index is not None
    ]
    return gb.GraphNode(
        id=_COMPUTE_NODE_ID,
        label="<compute>",
        attrs=[
            gb.KeyValue(key="entry point", value=entry_point.name),
            gb.KeyValue(
                key="execution model", value=entry_point.execution_model
            ),
            *_execution_mode_attrs(entry_point, model),
            gb.KeyValue(key="spv id", value=format_id(entry_point.id)),
            *disassembly_attrs(
                entry_point.instruction, disassembly_lines_by_byte_offset
            ),
        ],
        incomingEdges=[
            gb.IncomingEdge(
                sourceNodeId=_compute_resource_node_id(resource),
                sourceNodeOutputId="0",
                targetNodeInputId=str(resource.input_index),
            )
            for resource in input_resources
        ],
        inputsMetadata=[
            gb.MetadataItem(
                id=str(resource.input_index),
                attrs=[
                    gb.KeyValue(
                        key="descriptor",
                        value=_format_descriptor(
                            resource.descriptor_set, resource.binding
                        ),
                    )
                ],
            )
            for resource in input_resources
        ],
        outputsMetadata=[
            gb.MetadataItem(
                id=str(resource.output_index),
                attrs=[
                    gb.KeyValue(
                        key="descriptor",
                        value=_format_descriptor(
                            resource.descriptor_set, resource.binding
                        ),
                    )
                ],
            )
            for resource in output_resources
        ],
    )


def _execution_mode_attrs(
    entry_point: EntryPoint, model: SpirvModuleModel
) -> list[gb.KeyValue]:
    return [
        gb.KeyValue(
            key=f"execution mode {execution_mode.mode}",
            value=" ".join(
                str(operand) for operand in execution_mode.operands
            ),
        )
        for execution_mode in model.execution_modes
        if execution_mode.entry_point_id == entry_point.id
    ]


def _compute_group_attrs(
    entry_point: EntryPoint, resources: tuple[_ComputeResource, ...]
) -> dict[str, str]:
    return {
        "entry point": entry_point.name,
        "execution model": entry_point.execution_model,
        "input descriptors": str(
            sum(
                1 for resource in resources if resource.input_index is not None
            )
        ),
        "output descriptors": str(
            sum(
                1
                for resource in resources
                if resource.output_index is not None
            )
        ),
    }


def _compute_resource_node_id(resource: _ComputeResource) -> str:
    return f"resource_{resource.variable.id}"


def _compute_resource_label(
    resource: _ComputeResource, model: SpirvModuleModel
) -> str:
    name = model.debug_names.names.get(resource.variable.id)
    role = _compute_resource_role(resource)
    return f"{role} {name or format_id(resource.variable.id)}"


def _compute_resource_role(resource: _ComputeResource) -> str:
    if resource.input_index is not None and resource.output_index is not None:
        return "Input/Output"
    if resource.input_index is not None:
        return "Input"
    return "Output"


def _format_descriptor(descriptor_set: object, binding: object) -> str:
    return f"set {descriptor_set}, binding {binding}"


def _decoration_operand(
    decorations: Iterable[Decoration], decoration_name: str
) -> object | None:
    for decoration in decorations:
        if decoration.decoration == decoration_name and decoration.operands:
            return decoration.operands[0]
    return None


__all__ = ["build_compute_graph"]
