# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from model_explorer import graph_builder as gb

from .native.loader import SpirvToolsUnavailableError
from .native.spirv_tools import validate_bytes

VALIDATION_STATUS_KEY = "Validation"
VALIDATION_DIAGNOSTICS_KEY = "Validation diagnostics"
VALIDATION_ERROR_PREFIX = "SPIR-V validation failed before parsing"
VALIDATION_UNAVAILABLE_MESSAGE = "SPIRV-Tools validator is unavailable."


@dataclass(frozen=True)
class SpirvValidationDiagnostic:
    severity: str
    message: str
    position: int | None = None
    line: int | None = None
    column: int | None = None


@dataclass(frozen=True)
class SpirvValidationResult:
    ok: bool
    diagnostics: tuple[SpirvValidationDiagnostic, ...]
    result_code: int | None = None


def validate_spirv(
    data: bytes, *, target_env: str = "universal_1_6"
) -> SpirvValidationResult:
    """Validate SPIR-V bytes with the vendored SPIRV-Tools C API."""
    native_result = validate_bytes(data, target_env=target_env)
    severity = "info" if native_result.ok else "error"

    return SpirvValidationResult(
        ok=native_result.ok,
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


def validate_for_metadata(data: bytes) -> SpirvValidationResult | None:
    """Validate SPIR-V bytes, returning None when the validator is unavailable."""
    try:
        return validate_spirv(data)
    except SpirvToolsUnavailableError:
        return None


def attach_validation_metadata(
    graph: gb.Graph,
    validation_result: SpirvValidationResult | None,
    *,
    namespace: str = "",
) -> None:
    """Attach SPIR-V validation status to a Model Explorer graph."""
    group_attrs = graph.groupNodeAttributes or {}
    existing_graph_attrs = group_attrs.get(namespace, {})
    validation_attrs: dict[str, str] = {}
    if validation_result is None:
        validation_attrs[VALIDATION_STATUS_KEY] = "Unavailable"
        validation_attrs[VALIDATION_DIAGNOSTICS_KEY] = (
            VALIDATION_UNAVAILABLE_MESSAGE
        )
    elif validation_result.ok:
        validation_attrs[VALIDATION_STATUS_KEY] = "Passed"
    else:
        validation_attrs[VALIDATION_STATUS_KEY] = "Failed"
        validation_attrs[VALIDATION_DIAGNOSTICS_KEY] = (
            format_validation_result(validation_result)
        )
    group_attrs[namespace] = {**validation_attrs, **existing_graph_attrs}
    graph.groupNodeAttributes = group_attrs


def parsing_error_with_validation_context(
    err: Exception, validation_result: SpirvValidationResult | None
) -> ValueError:
    """Build a parse error that includes validation diagnostics when available."""
    message = f"Failed to parse SPIR-V binary: {err}"
    if validation_result is not None and not validation_result.ok:
        message = (
            f"{message}\n\n"
            f"{VALIDATION_ERROR_PREFIX}:\n"
            f"{format_validation_result(validation_result)}"
        )
    return ValueError(message)


def format_validation_result(validation_result: SpirvValidationResult) -> str:
    """Format all diagnostics for a validation result."""
    if validation_result.diagnostics:
        return "\n".join(
            format_validation_diagnostic(diagnostic)
            for diagnostic in validation_result.diagnostics
        )
    if validation_result.result_code is not None:
        return (
            f"Validator returned result code {validation_result.result_code}."
        )
    return "Validator reported failure without diagnostics."


def format_validation_diagnostic(
    diagnostic: SpirvValidationDiagnostic,
) -> str:
    """Format a single validation diagnostic."""
    location_parts = []
    if diagnostic.position is not None:
        location_parts.append(f"word {diagnostic.position}")
    if diagnostic.line is not None:
        location_parts.append(f"line {diagnostic.line}")
    if diagnostic.column is not None:
        location_parts.append(f"column {diagnostic.column}")

    location = f" ({', '.join(location_parts)})" if location_parts else ""
    return f"{diagnostic.severity}: {diagnostic.message}{location}"


__all__ = [
    "SpirvToolsUnavailableError",
    "SpirvValidationDiagnostic",
    "SpirvValidationResult",
    "attach_validation_metadata",
    "format_validation_diagnostic",
    "format_validation_result",
    "parsing_error_with_validation_context",
    "validate_for_metadata",
    "validate_spirv",
]
