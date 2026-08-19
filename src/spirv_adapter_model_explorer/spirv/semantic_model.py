# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .constants import _collect_constants, _resolve_constant_values
from .functions import _collect_functions
from .graphs import _collect_data_graph_entry_points, _collect_data_graphs
from .spirv_grammar import SpirvGrammar
from .spirv_parser import SpirvInstruction, SpirvModule
from .types import (
    ArrayType,
    DataGraph,
    DataGraphInput,
    DataGraphOperation,
    DataGraphOutput,
    DataGraphType,
    DebugNames,
    Decoration,
    DecorationIndex,
    Definition,
    EntryPoint,
    ExecutionMode,
    ExtInstImport,
    FunctionType,
    Id,
    MatrixType,
    MemberDecoration,
    MemoryModel,
    PointerType,
    RuntimeArrayType,
    ScalarType,
    SpirvModuleModel,
    StructType,
    TensorType,
    Type,
    UnknownType,
    Use,
    Variable,
    VectorType,
    _enum_display_name,
    _literal_int,
    _literal_string,
    _operand_value,
    _semantic_operands_after_result,
    _set_resolved_attr,
)


def build_semantic_model(
    module: SpirvModule, grammar: SpirvGrammar
) -> SpirvModuleModel:
    definitions = _collect_definitions(module)
    debug_names = _collect_debug_names(module)
    decorations = _collect_decorations(module)
    types = _collect_types(module)
    constants = _collect_constants(module)
    variables = _collect_variables(module)
    functions = _collect_functions(module)
    extended_instruction_imports = _collect_ext_inst_imports(module)
    data_graphs = _collect_data_graphs(
        module, grammar, extended_instruction_imports
    )
    uses = _collect_uses(module)

    entry_point_declarations = _collect_entry_point_declarations(module)

    model = SpirvModuleModel(
        header=module.header,
        capabilities=_collect_capabilities(module),
        extensions=_collect_extensions(module),
        extended_instruction_imports=extended_instruction_imports,
        memory_model=_collect_memory_model(module),
        entry_points=_entry_points_by_id(entry_point_declarations),
        entry_point_declarations=entry_point_declarations,
        data_graph_entry_points=_collect_data_graph_entry_points(module),
        execution_modes=_collect_execution_modes(module),
        debug_names=debug_names,
        decorations=decorations,
        types=types,
        constants=constants,
        variables=variables,
        functions=functions,
        data_graphs=data_graphs,
        global_instructions=_collect_global_instructions(module),
        definitions=definitions,
        uses=uses,
    )
    _resolve_references(model)
    _attach_graph_debug_info(model, module)
    return model


def _resolve_references(model: SpirvModuleModel) -> None:
    for type_model in model.types.values():
        _resolve_type_references(model, type_model)

    for constant in model.constants.values():
        _set_resolved_attr(constant, "type", model.types.get(constant.type_id))
    _resolve_constant_values(model)

    for variable in model.variables.values():
        _set_resolved_attr(variable, "type", model.types.get(variable.type_id))
        _set_resolved_attr(
            variable,
            "initializer",
            model.constants.get(variable.initializer_id)
            if variable.initializer_id is not None
            else None,
        )

    for function in model.functions.values():
        _set_resolved_attr(
            function, "return_type", model.types.get(function.return_type_id)
        )
        function_type = model.types.get(function.function_type_id)
        _set_resolved_attr(
            function,
            "function_type",
            function_type if isinstance(function_type, FunctionType) else None,
        )
        block_by_id = {block.id: block for block in function.blocks}
        for parameter in function.parameters:
            _set_resolved_attr(
                parameter, "type", model.types.get(parameter.type_id)
            )
        for block in function.blocks:
            _set_resolved_attr(
                block,
                "successor_blocks",
                tuple(
                    block_by_id.get(successor)
                    for successor in block.successors
                ),
            )

    for entry_point in model.entry_point_declarations:
        _set_resolved_attr(
            entry_point, "function", model.functions.get(entry_point.id)
        )
        _set_resolved_attr(
            entry_point,
            "interfaces",
            tuple(
                model.variables.get(interface_id)
                for interface_id in entry_point.interface_ids
            ),
        )

    entry_points_by_data_graph_id: dict[Id, list] = {}
    for data_graph_entry_point in model.data_graph_entry_points:
        entry_points_by_data_graph_id.setdefault(
            data_graph_entry_point.graph_id, []
        ).append(data_graph_entry_point)

    for graph in model.data_graphs.values():
        graph_type = model.types.get(graph.type_id)
        _set_resolved_attr(
            graph,
            "type",
            graph_type if isinstance(graph_type, DataGraphType) else None,
        )
        _set_resolved_attr(
            graph,
            "entry_points",
            tuple(entry_points_by_data_graph_id.get(graph.id, ())),
        )
        for graph_input in graph.inputs:
            _set_resolved_attr(
                graph_input, "type", model.types.get(graph_input.type_id)
            )
            _set_resolved_attr(
                graph_input,
                "input_index",
                model.constants.get(graph_input.input_index_id),
            )
            _set_resolved_attr(
                graph_input,
                "element_index",
                model.constants.get(graph_input.element_index_id)
                if graph_input.element_index_id is not None
                else None,
            )
        for operation in graph.operations:
            _set_resolved_attr(
                operation, "type", model.types.get(operation.type_id)
            )
            _set_resolved_attr(
                operation,
                "ext_inst_set",
                model.extended_instruction_imports.get(
                    operation.ext_inst_set_id
                )
                if operation.ext_inst_set_id is not None
                else None,
            )
            _set_resolved_attr(
                operation,
                "operands",
                tuple(
                    model.constants[operand_id].resolved_value
                    if operand_id in model.constants
                    else model.definitions.get(operand_id)
                    for operand_id in operation.operand_ids
                ),
            )
        for output in graph.outputs:
            _set_resolved_attr(
                output, "value", model.definitions.get(output.value_id)
            )
            _set_resolved_attr(
                output,
                "output_index",
                model.constants.get(output.output_index_id),
            )
            _set_resolved_attr(
                output,
                "element_index",
                model.constants.get(output.element_index_id)
                if output.element_index_id is not None
                else None,
            )

    for data_graph_entry_point in model.data_graph_entry_points:
        graph_definition = model.definitions.get(
            data_graph_entry_point.graph_id
        )
        _set_resolved_attr(
            data_graph_entry_point, "graph_definition", graph_definition
        )
        graph = model.data_graphs.get(data_graph_entry_point.graph_id)
        _set_resolved_attr(data_graph_entry_point, "graph", graph)
        _set_resolved_attr(
            data_graph_entry_point,
            "graph_type",
            graph.type if graph is not None else None,
        )
        _set_resolved_attr(
            data_graph_entry_point,
            "interfaces",
            tuple(
                model.variables.get(interface_id)
                for interface_id in data_graph_entry_point.interface_ids
            ),
        )

    for execution_mode in model.execution_modes:
        _set_resolved_attr(
            execution_mode,
            "entry_point",
            model.entry_points.get(execution_mode.entry_point_id),
        )


def _attach_graph_debug_info(
    model: SpirvModuleModel, module: SpirvModule
) -> None:
    graph_debug_set_ids = {
        id
        for id, ext_inst_import in model.extended_instruction_imports.items()
        if ext_inst_import.name.startswith("NonSemantic.Graph.DebugInfo.")
    }
    if not graph_debug_set_ids:
        return

    strings = _collect_debug_strings(module)
    data_graphs_by_id = model.data_graphs
    operations_by_id = {
        operation.id: operation
        for graph in model.data_graphs.values()
        for operation in graph.operations
    }
    inputs_by_id = {
        graph_input.id: graph_input
        for graph in model.data_graphs.values()
        for graph_input in graph.inputs
    }
    outputs_by_value_and_element = {
        (output.value_id, output.element_index_id): output
        for graph in model.data_graphs.values()
        for output in graph.outputs
    }
    output_debug_by_operation: dict[Id, dict[int, list[str]]] = {}

    for inst in module.instructions:
        if inst.opname != "OpExtInst" or len(inst.operands) < 5:
            continue
        ext_inst_set_id = inst.operands[2].ref_id
        if ext_inst_set_id not in graph_debug_set_ids:
            continue
        opcode = _literal_int(inst.operands[3].display)
        if opcode == 1:
            _attach_debug_graph(inst, strings, data_graphs_by_id)
        elif opcode == 2:
            _attach_debug_operation(inst, strings, operations_by_id)
        elif opcode == 3:
            _attach_debug_tensor(
                inst,
                strings,
                inputs_by_id,
                operations_by_id,
                outputs_by_value_and_element,
                output_debug_by_operation,
                model,
            )

    for operation_id, debug_by_output in output_debug_by_operation.items():
        operation = operations_by_id.get(operation_id)
        if operation is None:
            continue
        max_index = max(debug_by_output, default=-1)
        output_debug_info = [
            list(items) for items in operation.output_debug_info
        ]
        while len(output_debug_info) <= max_index:
            output_debug_info.append([])
        for output_index, items in debug_by_output.items():
            output_debug_info[output_index].extend(items)
        _set_resolved_attr(
            operation,
            "output_debug_info",
            tuple(
                tuple(_dedupe(debug_info)) for debug_info in output_debug_info
            ),
        )


def _collect_debug_strings(module: SpirvModule) -> dict[Id, str]:
    return {
        inst.result_id: _literal_string(inst.operands[1].display)
        for inst in module.instructions
        if inst.opname == "OpString"
        and inst.result_id is not None
        and len(inst.operands) >= 2
    }


def _attach_debug_graph(
    inst: SpirvInstruction,
    strings: dict[Id, str],
    data_graphs_by_id: dict[Id, DataGraph],
) -> None:
    graph_id = inst.operands[4].ref_id
    debug_info = _debug_string_operand(inst, strings, 5)
    if graph_id is None or debug_info is None:
        return
    graph = data_graphs_by_id.get(graph_id)
    if graph is not None:
        _append_debug_info(graph, debug_info)


def _attach_debug_operation(
    inst: SpirvInstruction,
    strings: dict[Id, str],
    operations_by_id: dict[Id, DataGraphOperation],
) -> None:
    debug_info = _debug_string_operand(inst, strings, 5)
    if debug_info is None:
        return
    for operand in inst.operands[6:]:
        target_id = operand.ref_id
        if target_id is None:
            continue
        operation = operations_by_id.get(target_id)
        if operation is not None:
            _append_debug_info(operation, debug_info)


def _attach_debug_tensor(
    inst: SpirvInstruction,
    strings: dict[Id, str],
    inputs_by_id: dict[Id, DataGraphInput],
    operations_by_id: dict[Id, DataGraphOperation],
    outputs_by_value_and_element: dict[tuple[Id, Id | None], DataGraphOutput],
    output_debug_by_operation: dict[Id, dict[int, list[str]]],
    model: SpirvModuleModel,
) -> None:
    tensor_id = inst.operands[4].ref_id
    debug_info = _debug_string_operand(inst, strings, 5)
    if tensor_id is None or debug_info is None:
        return
    element_index_id = (
        inst.operands[6].ref_id if len(inst.operands) >= 7 else None
    )

    output = outputs_by_value_and_element.get((tensor_id, element_index_id))
    if output is not None:
        _append_debug_info(output, debug_info)

    graph_input = inputs_by_id.get(tensor_id)
    if graph_input is not None and (
        element_index_id is None
        or graph_input.element_index_id == element_index_id
    ):
        if output is None:
            _append_debug_info(graph_input, debug_info)
        return

    operation = operations_by_id.get(tensor_id)
    if operation is not None:
        extract_selector = _composite_extract_output_selector(operation)
        if extract_selector is not None and element_index_id is None:
            source_id, output_index = extract_selector
            if source_id in operations_by_id:
                output_debug_by_operation.setdefault(source_id, {}).setdefault(
                    output_index, []
                ).append(debug_info)
                return

        output_index = (
            0
            if element_index_id is None
            else _constant_result_value(model, element_index_id)
        )
        if isinstance(output_index, int):
            output_debug_by_operation.setdefault(tensor_id, {}).setdefault(
                output_index, []
            ).append(debug_info)


def _composite_extract_output_selector(
    operation: object,
) -> tuple[Id, int] | None:
    instruction = getattr(operation, "instruction", None)
    if instruction is None or instruction.opname != "OpCompositeExtract":
        return None
    if len(instruction.operands) != 4:
        return None
    composite_id = instruction.operands[2].ref_id
    member_index = _literal_int(instruction.operands[3].display)
    if composite_id is None or member_index is None or member_index < 0:
        return None
    return composite_id, member_index


def _debug_string_operand(
    inst: SpirvInstruction, strings: dict[Id, str], operand_index: int
) -> str | None:
    if len(inst.operands) <= operand_index:
        return None
    string_id = inst.operands[operand_index].ref_id
    if string_id is None:
        return None
    return strings.get(string_id)


def _constant_result_value(model: SpirvModuleModel, constant_id: Id) -> object:
    constant = model.constants.get(constant_id)
    return constant_id if constant is None else constant.resolved_value


def _append_debug_info(target: object, debug_info: str) -> None:
    _set_resolved_attr(
        target,
        "debug_info",
        tuple(_dedupe((*getattr(target, "debug_info", ()), debug_info))),
    )


def _dedupe(items: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _resolve_type_references(
    model: SpirvModuleModel, type_model: Type
) -> None:
    if isinstance(type_model, VectorType):
        _set_resolved_attr(
            type_model,
            "component_type",
            model.types.get(type_model.component_type_id),
        )
    elif isinstance(type_model, MatrixType):
        _set_resolved_attr(
            type_model,
            "column_type",
            model.types.get(type_model.column_type_id),
        )
    elif isinstance(type_model, PointerType):
        _set_resolved_attr(
            type_model,
            "pointee_type",
            model.types.get(type_model.pointee_type_id),
        )
    elif isinstance(type_model, ArrayType):
        _set_resolved_attr(
            type_model,
            "element_type",
            model.types.get(type_model.element_type_id),
        )
        _set_resolved_attr(
            type_model, "length", model.constants.get(type_model.length_id)
        )
    elif isinstance(type_model, RuntimeArrayType):
        _set_resolved_attr(
            type_model,
            "element_type",
            model.types.get(type_model.element_type_id),
        )
    elif isinstance(type_model, StructType):
        _set_resolved_attr(
            type_model,
            "member_types",
            tuple(
                model.types.get(member_type_id)
                for member_type_id in type_model.member_type_ids
            ),
        )
    elif isinstance(type_model, FunctionType):
        _set_resolved_attr(
            type_model,
            "return_type",
            model.types.get(type_model.return_type_id),
        )
        _set_resolved_attr(
            type_model,
            "parameter_types",
            tuple(
                model.types.get(parameter_type_id)
                for parameter_type_id in type_model.parameter_type_ids
            ),
        )
    elif isinstance(type_model, TensorType):
        _set_resolved_attr(
            type_model,
            "element_type",
            model.types.get(type_model.element_type_id),
        )
        _set_resolved_attr(
            type_model, "rank", model.constants.get(type_model.rank_id)
        )
        _set_resolved_attr(
            type_model,
            "shape",
            model.constants.get(type_model.shape_id)
            if type_model.shape_id is not None
            else None,
        )
    elif isinstance(type_model, DataGraphType):
        _set_resolved_attr(
            type_model,
            "input_types",
            tuple(
                model.types.get(input_type_id)
                for input_type_id in type_model.input_type_ids
            ),
        )
        _set_resolved_attr(
            type_model,
            "output_types",
            tuple(
                model.types.get(output_type_id)
                for output_type_id in type_model.output_type_ids
            ),
        )


def _collect_definitions(module: SpirvModule) -> dict[Id, Definition]:
    return {
        inst.result_id: Definition(
            id=inst.result_id,
            instruction=inst,
            kind=_definition_kind(inst),
            type_id=inst.result_type_id,
        )
        for inst in module.instructions
        if inst.result_id is not None
    }


def _definition_kind(inst: SpirvInstruction) -> str:
    if inst.opname.startswith("OpType"):
        return "type"
    if inst.opname.startswith("OpConstant") or inst.opname.startswith(
        "OpSpecConstant"
    ):
        return "constant"
    if inst.opname == "OpVariable":
        return "variable"
    if inst.opname == "OpFunction":
        return "function"
    if inst.opname == "OpFunctionParameter":
        return "function_parameter"
    if inst.opname == "OpGraphARM":
        return "graph"
    if inst.opname == "OpLabel":
        return "basic_block"
    if inst.opname == "OpExtInstImport":
        return "extended_instruction_import"
    if inst.opname == "OpDecorationGroup":
        return "decoration_group"
    if inst.opname == "OpString":
        return "debug_string"
    return "instruction"


def _collect_uses(module: SpirvModule) -> dict[Id, tuple[Use, ...]]:
    uses: dict[Id, list[Use]] = {}
    for inst in module.instructions:
        for index, operand in enumerate(inst.operands):
            if operand.ref_id is None:
                continue
            uses.setdefault(operand.ref_id, []).append(
                Use(
                    used_id=operand.ref_id,
                    user_instruction=inst,
                    operand_index=index,
                    role=operand.name,
                )
            )
    return {used_id: tuple(id_uses) for used_id, id_uses in uses.items()}


def _collect_debug_names(module: SpirvModule) -> DebugNames:
    names: dict[Id, str] = {}
    member_names: dict[tuple[Id, int], str] = {}
    for inst in module.instructions:
        if inst.opname == "OpName" and len(inst.operands) >= 2:
            target = inst.operands[0].ref_id
            if target is not None:
                names[target] = _literal_string(inst.operands[1].display)
        elif inst.opname == "OpMemberName" and len(inst.operands) >= 3:
            target = inst.operands[0].ref_id
            member_index = _literal_int(inst.operands[1].display)
            if target is not None and member_index is not None:
                member_names[(target, member_index)] = _literal_string(
                    inst.operands[2].display
                )
    return DebugNames(names=names, member_names=member_names)


def _collect_decorations(module: SpirvModule) -> DecorationIndex:
    direct: dict[Id, list[Decoration]] = {}
    member: dict[tuple[Id, int], list[MemberDecoration]] = {}
    group_decorations: dict[Id, list[Decoration]] = {}
    group_member_decorations: dict[Id, list[MemberDecoration]] = {}

    for inst in module.instructions:
        if inst.opname == "OpDecorate" and len(inst.operands) >= 2:
            target = inst.operands[0].ref_id
            if target is None:
                continue
            decoration = Decoration(
                target_id=target,
                decoration=_enum_display_name(inst.operands[1].display),
                operands=tuple(
                    _operand_value(operand) for operand in inst.operands[2:]
                ),
                instruction=inst,
            )
            if _definition_kind_for_id(module, target) == "decoration_group":
                group_decorations.setdefault(target, []).append(decoration)
            else:
                direct.setdefault(target, []).append(decoration)
        elif inst.opname == "OpMemberDecorate" and len(inst.operands) >= 3:
            target = inst.operands[0].ref_id
            member_index = _literal_int(inst.operands[1].display)
            if target is None or member_index is None:
                continue
            decoration = MemberDecoration(
                target_id=target,
                member_index=member_index,
                decoration=_enum_display_name(inst.operands[2].display),
                operands=tuple(
                    _operand_value(operand) for operand in inst.operands[3:]
                ),
                instruction=inst,
            )
            if _definition_kind_for_id(module, target) == "decoration_group":
                group_member_decorations.setdefault(target, []).append(
                    decoration
                )
            else:
                member.setdefault((target, member_index), []).append(
                    decoration
                )

    for inst in module.instructions:
        if inst.opname == "OpGroupDecorate" and inst.operands:
            group_id = inst.operands[0].ref_id
            if group_id is None:
                continue
            for operand in inst.operands[1:]:
                if operand.ref_id is None:
                    continue
                for decoration in group_decorations.get(group_id, []):
                    direct.setdefault(operand.ref_id, []).append(
                        Decoration(
                            target_id=operand.ref_id,
                            decoration=decoration.decoration,
                            operands=decoration.operands,
                            instruction=inst,
                        )
                    )
        elif inst.opname == "OpGroupMemberDecorate" and inst.operands:
            group_id = inst.operands[0].ref_id
            if group_id is None:
                continue
            idx = 1
            while idx + 1 < len(inst.operands):
                target = inst.operands[idx].ref_id
                member_index = _literal_int(inst.operands[idx + 1].display)
                if target is not None and member_index is not None:
                    for decoration in group_member_decorations.get(
                        group_id, []
                    ):
                        member.setdefault((target, member_index), []).append(
                            MemberDecoration(
                                target_id=target,
                                member_index=member_index,
                                decoration=decoration.decoration,
                                operands=decoration.operands,
                                instruction=inst,
                            )
                        )
                idx += 2

    return DecorationIndex(
        by_target={target: tuple(items) for target, items in direct.items()},
        by_member={target: tuple(items) for target, items in member.items()},
    )


def _definition_kind_for_id(module: SpirvModule, id: Id) -> str | None:
    for inst in module.instructions:
        if inst.result_id == id:
            if inst.opname == "OpDecorationGroup":
                return "decoration_group"
            return _definition_kind(inst)
    return None


def _collect_types(module: SpirvModule) -> dict[Id, Type]:
    types: dict[Id, Type] = {}
    for inst in module.instructions:
        if inst.result_id is None or not inst.opname.startswith("OpType"):
            continue
        values = _semantic_operands_after_result(inst)
        if inst.opname == "OpTypeVoid":
            type_model: Type = ScalarType(
                inst.result_id, inst.opname, inst, "void"
            )
        elif inst.opname == "OpTypeBool":
            type_model = ScalarType(inst.result_id, inst.opname, inst, "bool")
        elif inst.opname == "OpTypeInt" and len(values) >= 2:
            type_model = ScalarType(
                inst.result_id,
                inst.opname,
                inst,
                "int",
                width=int(values[0]),
                signed=bool(values[1]),
            )
        elif inst.opname == "OpTypeFloat" and values:
            type_model = ScalarType(
                inst.result_id,
                inst.opname,
                inst,
                "float",
                width=int(values[0]),
            )
        elif inst.opname == "OpTypeVector" and len(values) >= 2:
            type_model = VectorType(
                inst.result_id,
                inst.opname,
                inst,
                int(values[0]),
                int(values[1]),
            )
        elif inst.opname == "OpTypeMatrix" and len(values) >= 2:
            type_model = MatrixType(
                inst.result_id,
                inst.opname,
                inst,
                int(values[0]),
                int(values[1]),
            )
        elif inst.opname == "OpTypePointer" and len(values) >= 2:
            type_model = PointerType(
                inst.result_id,
                inst.opname,
                inst,
                str(values[0]),
                int(values[1]),
            )
        elif inst.opname == "OpTypeArray" and len(values) >= 2:
            type_model = ArrayType(
                inst.result_id,
                inst.opname,
                inst,
                int(values[0]),
                int(values[1]),
            )
        elif inst.opname == "OpTypeRuntimeArray" and values:
            type_model = RuntimeArrayType(
                inst.result_id, inst.opname, inst, int(values[0])
            )
        elif inst.opname == "OpTypeStruct":
            type_model = StructType(
                inst.result_id,
                inst.opname,
                inst,
                tuple(int(value) for value in values),
            )
        elif inst.opname == "OpTypeFunction" and values:
            type_model = FunctionType(
                inst.result_id,
                inst.opname,
                inst,
                return_type_id=int(values[0]),
                parameter_type_ids=tuple(int(value) for value in values[1:]),
            )
        elif inst.opname == "OpTypeTensorARM" and len(values) >= 2:
            type_model = TensorType(
                inst.result_id,
                inst.opname,
                inst,
                element_type_id=int(values[0]),
                rank_id=int(values[1]),
                shape_id=int(values[2]) if len(values) >= 3 else None,
            )
        elif inst.opname == "OpTypeGraphARM" and values:
            num_inputs = int(values[0])
            in_out_type_ids = tuple(int(value) for value in values[1:])
            type_model = DataGraphType(
                inst.result_id,
                inst.opname,
                inst,
                input_type_ids=in_out_type_ids[:num_inputs],
                output_type_ids=in_out_type_ids[num_inputs:],
            )
        else:
            type_model = UnknownType(
                inst.result_id, inst.opname, inst, operands=tuple(values)
            )
        types[inst.result_id] = type_model
    return types


def _collect_variables(module: SpirvModule) -> dict[Id, Variable]:
    variables: dict[Id, Variable] = {}
    for inst in module.instructions:
        if (
            inst.opname != "OpVariable"
            or inst.result_id is None
            or inst.result_type_id is None
            or len(inst.operands) < 3
        ):
            continue
        initializer = next(
            (
                operand.ref_id
                for operand in inst.operands[3:]
                if operand.ref_id is not None
            ),
            None,
        )
        variables[inst.result_id] = Variable(
            id=inst.result_id,
            type_id=inst.result_type_id,
            storage_class=_enum_display_name(inst.operands[2].display),
            initializer_id=initializer,
            instruction=inst,
        )
    return variables


def _collect_capabilities(module: SpirvModule) -> tuple[str, ...]:
    return tuple(
        _enum_display_name(inst.operands[0].display)
        for inst in module.instructions
        if inst.opname == "OpCapability" and inst.operands
    )


def _collect_extensions(module: SpirvModule) -> tuple[str, ...]:
    return tuple(
        _literal_string(inst.operands[0].display)
        for inst in module.instructions
        if inst.opname == "OpExtension" and inst.operands
    )


def _collect_ext_inst_imports(module: SpirvModule) -> dict[Id, ExtInstImport]:
    return {
        inst.result_id: ExtInstImport(
            id=inst.result_id,
            name=_literal_string(inst.operands[1].display),
            instruction=inst,
        )
        for inst in module.instructions
        if inst.opname == "OpExtInstImport"
        and inst.result_id is not None
        and len(inst.operands) >= 2
    }


def _collect_memory_model(module: SpirvModule) -> MemoryModel | None:
    for inst in module.instructions:
        if inst.opname == "OpMemoryModel" and len(inst.operands) >= 2:
            return MemoryModel(
                addressing_model=_enum_display_name(inst.operands[0].display),
                memory_model=_enum_display_name(inst.operands[1].display),
                instruction=inst,
            )
    return None


def _collect_entry_point_declarations(
    module: SpirvModule,
) -> tuple[EntryPoint, ...]:
    entry_points: list[EntryPoint] = []
    for inst in module.instructions:
        if inst.opname != "OpEntryPoint" or len(inst.operands) < 3:
            continue
        entry_id = inst.operands[1].ref_id
        if entry_id is None:
            continue
        entry_points.append(
            EntryPoint(
                id=entry_id,
                execution_model=_enum_display_name(inst.operands[0].display),
                name=_literal_string(inst.operands[2].display),
                interface_ids=tuple(
                    operand.ref_id
                    for operand in inst.operands[3:]
                    if operand.ref_id is not None
                ),
                instruction=inst,
            ),
        )
    return tuple(entry_points)


def _entry_points_by_id(
    entry_point_declarations: tuple[EntryPoint, ...],
) -> dict[Id, EntryPoint]:
    return {
        entry_point.id: entry_point for entry_point in entry_point_declarations
    }


def _collect_execution_modes(module: SpirvModule) -> tuple[ExecutionMode, ...]:
    modes: list[ExecutionMode] = []
    for inst in module.instructions:
        if (
            inst.opname not in {"OpExecutionMode", "OpExecutionModeId"}
            or len(inst.operands) < 2
        ):
            continue
        entry_id = inst.operands[0].ref_id
        if entry_id is None:
            continue
        modes.append(
            ExecutionMode(
                entry_point_id=entry_id,
                mode=_enum_display_name(inst.operands[1].display),
                operands=tuple(
                    _operand_value(operand) for operand in inst.operands[2:]
                ),
                instruction=inst,
            )
        )
    return tuple(modes)


def _collect_global_instructions(
    module: SpirvModule,
) -> tuple[SpirvInstruction, ...]:
    global_instructions: list[SpirvInstruction] = []
    in_function = False
    for inst in module.instructions:
        if inst.opname == "OpFunction":
            in_function = True
        if not in_function:
            global_instructions.append(inst)
        if inst.opname == "OpFunctionEnd":
            in_function = False
    return tuple(global_instructions)


__all__ = ["build_semantic_model"]
