# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from spirv_adapter_model_explorer.assembly import (
    assemble_spirv_text,
    disassemble_spirv,
)

from .conftest import fixture_text
from .test_spirv_parser import assemble_valid_spirv

VALID_ASM = fixture_text("basic-compute.spvasm")


def test_assemble_spirv_text_returns_binary():
    result = assemble_spirv_text(VALID_ASM)

    assert result.ok
    assert result.data[:4] == b"\x03\x02\x23\x07"
    assert result.diagnostics == ()


def test_assemble_spirv_text_returns_diagnostics_for_invalid_text():
    result = assemble_spirv_text("OpDefinitelyNotAnInstruction\n")

    assert not result.ok
    assert result.data == b""
    assert result.diagnostics
    assert result.diagnostics[0].message


def test_disassemble_spirv_returns_text():
    binary = assemble_valid_spirv(VALID_ASM)

    result = disassemble_spirv(binary)

    assert result.ok
    assert "OpEntryPoint GLCompute" in result.text
    assert "OpFunction" in result.text
    assert result.diagnostics == ()


def test_disassemble_spirv_can_map_lines_by_byte_offset():
    binary = assemble_valid_spirv(VALID_ASM)

    result = disassemble_spirv(binary, show_byte_offset=True)

    assert result.ok
    assert result.lines_by_byte_offset is not None
    entry_point_line = result.lines_by_byte_offset[40]
    assert entry_point_line.line_number > 0
    assert entry_point_line.byte_offset == 40
    assert "OpEntryPoint GLCompute" in entry_point_line.text
    assert entry_point_line.text.rstrip().endswith("; 0x00000028")


def test_disassemble_spirv_rejects_non_word_aligned_binary():
    result = disassemble_spirv(b"abc")

    assert not result.ok
    assert result.text == ""
    assert result.diagnostics
    assert result.diagnostics[0].position is None
