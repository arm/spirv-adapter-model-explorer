# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
from dataclasses import dataclass

from .spirv_grammar import InstructionSpec, OperandSpec, SpirvGrammar

SPIRV_MAGIC = 0x07230203


@dataclass(frozen=True)
class SpirvHeader:
    magic: int
    version: int
    generator: int
    bound: int
    schema: int


@dataclass(frozen=True)
class SpirvOperand:
    kind: str
    display: str
    ref_id: int | None = None
    name: str | None = None


@dataclass(frozen=True)
class SpirvInstruction:
    index: int
    word_offset: int
    word_count: int
    opcode: int
    opname: str
    operands: tuple[SpirvOperand, ...]
    raw_words: tuple[int, ...]
    result_type_id: int | None = None
    result_id: int | None = None


@dataclass(frozen=True)
class SpirvModule:
    header: SpirvHeader
    instructions: tuple[SpirvInstruction, ...]


def _decode_c_string_from_words(
    words: tuple[int, ...], start: int
) -> tuple[str, int]:
    data = bytearray()
    idx = start
    while idx < len(words):
        data.extend(struct.pack("<I", words[idx]))
        idx += 1
        if 0 in data[-4:]:
            break

    nul = data.find(0)
    if nul != -1:
        data = data[:nul]

    return data.decode("utf-8", errors="replace"), idx - start


def _format_id(value: int) -> str:
    return f"%{value}"


def _operand_is_id_ref(kind: str) -> bool:
    return kind.startswith("Id") and kind not in {"IdResult", "IdResultType"}


def _decode_single_operand(
    operand_words: tuple[int, ...],
    word_index: int,
    spec: OperandSpec,
    grammar: SpirvGrammar,
) -> tuple[SpirvOperand, int]:
    kind = spec.kind
    name = spec.name

    if kind == "LiteralString":
        value, consumed = _decode_c_string_from_words(
            operand_words, word_index
        )
        return SpirvOperand(
            kind=kind, display=f'"{value}"', name=name
        ), consumed

    value = operand_words[word_index]

    if _operand_is_id_ref(kind):
        return (
            SpirvOperand(
                kind=kind,
                display=_format_id(value),
                ref_id=value,
                name=name,
            ),
            1,
        )

    if kind in {"IdResult", "IdResultType"}:
        return (
            SpirvOperand(kind=kind, display=_format_id(value), name=name),
            1,
        )

    if kind.startswith("Literal"):
        return SpirvOperand(kind=kind, display=str(value), name=name), 1

    enum_name = grammar.enum_name(kind, int(value))
    if enum_name is not None:
        return (
            SpirvOperand(
                kind=kind, display=f"{enum_name}({value})", name=name
            ),
            1,
        )

    return SpirvOperand(kind=kind, display=f"0x{value:08x}", name=name), 1


def _decode_operands(
    operand_words: tuple[int, ...],
    spec: InstructionSpec,
    grammar: SpirvGrammar,
) -> tuple[SpirvOperand, ...]:
    operands: list[SpirvOperand] = []
    idx = 0

    for operand_spec in spec.operands:
        if operand_spec.quantifier == "?":
            if idx >= len(operand_words):
                continue
            operand, consumed = _decode_single_operand(
                operand_words, idx, operand_spec, grammar
            )
            operands.append(operand)
            idx += consumed
            continue

        if operand_spec.quantifier == "*":
            while idx < len(operand_words):
                operand, consumed = _decode_single_operand(
                    operand_words, idx, operand_spec, grammar
                )
                operands.append(operand)
                idx += consumed
            continue

        if idx >= len(operand_words):
            break

        operand, consumed = _decode_single_operand(
            operand_words, idx, operand_spec, grammar
        )
        operands.append(operand)
        idx += consumed

    while idx < len(operand_words):
        value = operand_words[idx]
        operands.append(
            SpirvOperand(kind="(unknown)", display=f"0x{value:08x}")
        )
        idx += 1

    return tuple(operands)


def _find_result_ids(
    operands: tuple[SpirvOperand, ...],
) -> tuple[int | None, int | None]:
    result_type_id = None
    result_id = None

    for operand in operands:
        if operand.kind == "IdResultType":
            result_type_id = int(operand.display[1:])
        elif operand.kind == "IdResult":
            result_id = int(operand.display[1:])

    return result_type_id, result_id


def parse_spirv(data: bytes, grammar: SpirvGrammar) -> SpirvModule:
    if len(data) % 4 != 0:
        raise ValueError("Invalid SPIR-V: byte length is not a multiple of 4")

    words = struct.unpack("<" + "I" * (len(data) // 4), data)
    if len(words) < 5:
        raise ValueError("Invalid SPIR-V: missing header")

    header = SpirvHeader(
        magic=words[0],
        version=words[1],
        generator=words[2],
        bound=words[3],
        schema=words[4],
    )
    if header.magic != SPIRV_MAGIC:
        raise ValueError(f"Invalid SPIR-V magic: got 0x{header.magic:08x}")

    instructions: list[SpirvInstruction] = []
    word_offset = 5
    inst_index = 0

    while word_offset < len(words):
        first_word = words[word_offset]
        word_count = (first_word >> 16) & 0xFFFF
        opcode = first_word & 0xFFFF

        if word_count == 0:
            raise ValueError(
                f"Invalid SPIR-V: word_count=0 at word offset {word_offset}"
            )

        end = word_offset + word_count
        if end > len(words):
            raise ValueError(
                "Invalid SPIR-V: instruction overruns module bounds at "
                f"word offset {word_offset}"
            )

        raw_words = words[word_offset:end]
        operand_words = raw_words[1:]
        spec = grammar.instruction(opcode)

        if spec is None:
            opname = f"Op({opcode})"
            operands = tuple(
                SpirvOperand(kind="word", display=f"0x{word:08x}")
                for word in operand_words
            )
            result_type_id = None
            result_id = None
        else:
            opname = spec.opname
            operands = _decode_operands(operand_words, spec, grammar)
            result_type_id, result_id = _find_result_ids(operands)

        instructions.append(
            SpirvInstruction(
                index=inst_index,
                word_offset=word_offset,
                word_count=word_count,
                opcode=opcode,
                opname=opname,
                operands=operands,
                raw_words=raw_words,
                result_type_id=result_type_id,
                result_id=result_id,
            )
        )

        inst_index += 1
        word_offset = end

    return SpirvModule(header=header, instructions=tuple(instructions))
