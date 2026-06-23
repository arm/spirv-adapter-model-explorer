# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.resources
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

_EXTINST_SET_TO_RESOURCE_NAME: dict[str, str] = {
    "TOSA.001000.1": "extinst.tosa.001000.1.grammar.json",
}


@dataclass(frozen=True)
class OperandSpec:
    kind: str
    name: str | None = None
    quantifier: str | None = None


@dataclass(frozen=True)
class InstructionSpec:
    opname: str
    opcode: int
    operands: tuple[OperandSpec, ...]


@dataclass(frozen=True)
class EnumSpec:
    kind: str
    value_to_name: dict[int, str]


@dataclass(frozen=True)
class SpirvGrammar:
    instructions_by_opcode: dict[int, InstructionSpec]
    enums_by_kind: dict[str, EnumSpec]
    extinst_by_set_and_opcode: dict[str, dict[int, InstructionSpec]]

    def instruction(self, opcode: int) -> InstructionSpec | None:
        return self.instructions_by_opcode.get(opcode)

    def enum_name(self, kind: str, value: int) -> str | None:
        spec = self.enums_by_kind.get(kind)
        if spec is None:
            return None
        return spec.value_to_name.get(value)

    def extinst_instruction_spec(
        self, instruction_set: str, opcode: int
    ) -> InstructionSpec | None:
        return self.extinst_by_set_and_opcode.get(instruction_set, {}).get(
            opcode
        )


def _resource_json(resource_name: str) -> dict[str, Any]:
    path = importlib.resources.files(
        "spirv_adapter_model_explorer.resources"
    ).joinpath(resource_name)
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_instruction(inst: dict[str, Any]) -> InstructionSpec:
    opcode = int(inst["opcode"])
    operands = tuple(
        OperandSpec(
            kind=operand["kind"],
            name=operand.get("name"),
            quantifier=operand.get("quantifier"),
        )
        for operand in inst.get("operands", [])
    )
    return InstructionSpec(
        opname=inst["opname"],
        opcode=opcode,
        operands=operands,
    )


def _parse_extinst_instruction(inst: dict[str, Any]) -> InstructionSpec | None:
    try:
        opcode = int(inst["opcode"])
    except (KeyError, TypeError, ValueError):
        return None

    opname = inst.get("opname")
    if not opname:
        return None

    operands = tuple(
        OperandSpec(
            kind=operand.get("kind", "IdRef"),
            name=operand.get("name"),
            quantifier=operand.get("quantifier"),
        )
        for operand in inst.get("operands", [])
    )
    return InstructionSpec(opname=opname, opcode=opcode, operands=operands)


@lru_cache(maxsize=1)
def load_core_grammar() -> SpirvGrammar:
    data = _resource_json("spirv.core.grammar.json")

    instructions_by_opcode = {
        spec.opcode: spec
        for inst in data.get("instructions", [])
        for spec in [_parse_instruction(inst)]
    }

    enums_by_kind: dict[str, EnumSpec] = {}
    for kind in data.get("operand_kinds", []):
        if kind.get("category") not in {"ValueEnum", "BitEnum"}:
            continue

        value_to_name: dict[int, str] = {}
        for enumerant in kind.get("enumerants", []):
            try:
                value = int(enumerant["value"])
            except (KeyError, TypeError, ValueError):
                continue
            value_to_name[value] = enumerant.get("enumerant", str(value))

        enums_by_kind[kind["kind"]] = EnumSpec(
            kind=kind["kind"],
            value_to_name=value_to_name,
        )

    extinst_by_set_and_opcode: dict[str, dict[int, InstructionSpec]] = {}
    for (
        instruction_set,
        resource_name,
    ) in _EXTINST_SET_TO_RESOURCE_NAME.items():
        extinst_data = _resource_json(resource_name)
        extinst_by_set_and_opcode[instruction_set] = {
            spec.opcode: spec
            for inst in extinst_data.get("instructions", [])
            if (spec := _parse_extinst_instruction(inst)) is not None
        }

    return SpirvGrammar(
        instructions_by_opcode=instructions_by_opcode,
        enums_by_kind=enums_by_kind,
        extinst_by_set_and_opcode=extinst_by_set_and_opcode,
    )
