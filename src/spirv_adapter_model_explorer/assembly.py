# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass

from .native.loader import SpirvToolsUnavailableError
from .native.spirv_tools import assemble_text, disassemble_bytes
from .validation import SpirvValidationDiagnostic

_OFFSET_COMMENT_RE = re.compile(r"(?:^|;)\s*(0x[0-9a-fA-F]+)\s*$")


@dataclass(frozen=True)
class SpirvAssemblyResult:
    ok: bool
    data: bytes
    diagnostics: tuple[SpirvValidationDiagnostic, ...]
    result_code: int | None = None


@dataclass(frozen=True)
class SpirvDisassemblyLine:
    line_number: int
    byte_offset: int
    text: str


@dataclass(frozen=True)
class SpirvDisassemblyResult:
    ok: bool
    text: str
    diagnostics: tuple[SpirvValidationDiagnostic, ...]
    result_code: int | None = None
    lines_by_byte_offset: dict[int, SpirvDisassemblyLine] | None = None


def assemble_spirv_text(
    text: str,
    *,
    target_env: str = "universal_1_0",
    preserve_numeric_ids: bool = True,
) -> SpirvAssemblyResult:
    """Assemble SPIR-V text to bytes with the vendored SPIRV-Tools C API."""
    native_result = assemble_text(
        text,
        target_env=target_env,
        preserve_numeric_ids=preserve_numeric_ids,
    )
    severity = "info" if native_result.ok else "error"
    return SpirvAssemblyResult(
        ok=native_result.ok,
        data=native_result.data,
        result_code=native_result.result_code,
        diagnostics=tuple(
            SpirvValidationDiagnostic(
                severity=severity,
                message=diagnostic.message,
                position=diagnostic.position,
                line=diagnostic.line,
                column=diagnostic.column,
            )
            for diagnostic in native_result.diagnostics
        ),
    )


def disassemble_spirv(
    data: bytes,
    *,
    target_env: str = "universal_1_0",
    friendly_names: bool = True,
    indent: bool = True,
    show_byte_offset: bool = False,
) -> SpirvDisassemblyResult:
    """Disassemble SPIR-V bytes to text with the vendored SPIRV-Tools C API."""
    native_result = disassemble_bytes(
        data,
        target_env=target_env,
        friendly_names=friendly_names,
        indent=indent,
        show_byte_offset=show_byte_offset,
    )
    severity = "info" if native_result.ok else "error"
    return SpirvDisassemblyResult(
        ok=native_result.ok,
        text=native_result.text,
        result_code=native_result.result_code,
        lines_by_byte_offset=_lines_by_byte_offset(native_result.text)
        if native_result.ok and show_byte_offset
        else None,
        diagnostics=tuple(
            SpirvValidationDiagnostic(
                severity=severity,
                message=diagnostic.message,
                position=diagnostic.position,
                line=diagnostic.line,
                column=diagnostic.column,
            )
            for diagnostic in native_result.diagnostics
        ),
    )


def _lines_by_byte_offset(text: str) -> dict[int, SpirvDisassemblyLine]:
    lines: dict[int, SpirvDisassemblyLine] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _OFFSET_COMMENT_RE.search(line)
        if match is None:
            continue
        byte_offset = int(match.group(1), 16)
        lines[byte_offset] = SpirvDisassemblyLine(
            line_number=line_number,
            byte_offset=byte_offset,
            text=line,
        )
    return lines


__all__ = [
    "SpirvAssemblyResult",
    "SpirvDisassemblyLine",
    "SpirvDisassemblyResult",
    "SpirvToolsUnavailableError",
    "assemble_spirv_text",
    "disassemble_spirv",
]
