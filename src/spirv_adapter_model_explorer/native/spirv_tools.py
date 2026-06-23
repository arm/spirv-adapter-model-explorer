# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from cffi import FFI

from .loader import spirv_tools_library_path
from .spirv_tools_cdef import CDEF


@dataclass(frozen=True)
class NativeDiagnostic:
    message: str
    position: int | None
    line: int | None
    column: int | None


@dataclass(frozen=True)
class NativeAssemblyResult:
    ok: bool
    data: bytes
    result_code: int
    diagnostics: tuple[NativeDiagnostic, ...]


@dataclass(frozen=True)
class NativeDisassemblyResult:
    ok: bool
    text: str
    result_code: int
    diagnostics: tuple[NativeDiagnostic, ...]


@dataclass(frozen=True)
class NativeValidationResult:
    ok: bool
    result_code: int
    diagnostics: tuple[NativeDiagnostic, ...]


TARGET_ENVS = {
    "universal_1_0": "SPV_ENV_UNIVERSAL_1_0",
    "universal_1_1": "SPV_ENV_UNIVERSAL_1_1",
    "universal_1_2": "SPV_ENV_UNIVERSAL_1_2",
    "universal_1_3": "SPV_ENV_UNIVERSAL_1_3",
    "universal_1_4": "SPV_ENV_UNIVERSAL_1_4",
    "universal_1_5": "SPV_ENV_UNIVERSAL_1_5",
    "universal_1_6": "SPV_ENV_UNIVERSAL_1_6",
    "vulkan_1_0": "SPV_ENV_VULKAN_1_0",
    "vulkan_1_1": "SPV_ENV_VULKAN_1_1",
    "vulkan_1_1_spirv_1_4": "SPV_ENV_VULKAN_1_1_SPIRV_1_4",
    "vulkan_1_2": "SPV_ENV_VULKAN_1_2",
    "vulkan_1_3": "SPV_ENV_VULKAN_1_3",
    "vulkan_1_4": "SPV_ENV_VULKAN_1_4",
}

_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS = 1 << 1
_BINARY_TO_TEXT_OPTION_INDENT = 1 << 3
_BINARY_TO_TEXT_OPTION_SHOW_BYTE_OFFSET = 1 << 4
_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES = 1 << 6


@lru_cache(maxsize=1)
def _ffi_and_lib():
    ffi = FFI()
    ffi.cdef(CDEF)
    return ffi, ffi.dlopen(spirv_tools_library_path())


def _target_env_value(target_env: str):
    _ffi, lib = _ffi_and_lib()

    try:
        return getattr(lib, TARGET_ENVS[target_env])
    except KeyError as err:
        supported = ", ".join(sorted(TARGET_ENVS))
        raise ValueError(
            f"Unsupported SPIR-V target environment {target_env!r}. "
            f"Supported values: {supported}"
        ) from err


def _create_context(ffi, lib, target_env: str):
    context = lib.spvContextCreate(_target_env_value(target_env))
    if context == ffi.NULL:
        raise RuntimeError(
            f"spvContextCreate failed for target environment {target_env!r}"
        )
    return context


def _diagnostic_from_native(ffi, diagnostic) -> NativeDiagnostic:
    message = ""
    if diagnostic.error != ffi.NULL:
        message = ffi.string(diagnostic.error).decode(
            "utf-8", errors="replace"
        )

    return NativeDiagnostic(
        message=message,
        position=int(diagnostic.position.index),
        line=int(diagnostic.position.line),
        column=int(diagnostic.position.column),
    )


def _diagnostics_from_out(ffi, diagnostic_out) -> tuple[NativeDiagnostic, ...]:
    if diagnostic_out[0] == ffi.NULL:
        return ()
    return (_diagnostic_from_native(ffi, diagnostic_out[0]),)


def _non_word_aligned_diagnostic() -> NativeDiagnostic:
    return NativeDiagnostic(
        message="SPIR-V binary length is not a multiple of 4 bytes",
        position=None,
        line=None,
        column=None,
    )


def assemble_text(
    text: str,
    *,
    target_env: str = "universal_1_6",
    preserve_numeric_ids: bool = True,
) -> NativeAssemblyResult:
    ffi, lib = _ffi_and_lib()
    context = _create_context(ffi, lib, target_env)
    diagnostic_out = ffi.new("spv_diagnostic*")
    binary_out = ffi.new("spv_binary*")
    encoded = text.encode("utf-8")
    text_buffer = ffi.from_buffer("const char[]", encoded)
    options = (
        _TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS
        if preserve_numeric_ids
        else 0
    )

    try:
        result_code = int(
            lib.spvTextToBinaryWithOptions(
                context,
                text_buffer,
                len(encoded),
                options,
                binary_out,
                diagnostic_out,
            )
        )
        data = b""
        if binary_out[0] != ffi.NULL:
            data = bytes(
                ffi.buffer(
                    binary_out[0].code,
                    int(binary_out[0].wordCount) * ffi.sizeof("uint32_t"),
                )
            )
        return NativeAssemblyResult(
            ok=result_code == int(lib.SPV_SUCCESS),
            data=data,
            result_code=result_code,
            diagnostics=_diagnostics_from_out(ffi, diagnostic_out),
        )
    finally:
        if binary_out[0] != ffi.NULL:
            lib.spvBinaryDestroy(binary_out[0])
        if diagnostic_out[0] != ffi.NULL:
            lib.spvDiagnosticDestroy(diagnostic_out[0])
        lib.spvContextDestroy(context)


def disassemble_bytes(
    data: bytes,
    *,
    target_env: str = "universal_1_6",
    friendly_names: bool = True,
    indent: bool = True,
    show_byte_offset: bool = False,
) -> NativeDisassemblyResult:
    if len(data) % 4 != 0:
        return NativeDisassemblyResult(
            ok=False,
            text="",
            result_code=-4,
            diagnostics=(_non_word_aligned_diagnostic(),),
        )

    ffi, lib = _ffi_and_lib()
    context = _create_context(ffi, lib, target_env)
    diagnostic_out = ffi.new("spv_diagnostic*")
    text_out = ffi.new("spv_text*")
    words = ffi.from_buffer("const uint32_t[]", data)
    options = 0
    if friendly_names:
        options |= _BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES
    if indent:
        options |= _BINARY_TO_TEXT_OPTION_INDENT
    if show_byte_offset:
        options |= _BINARY_TO_TEXT_OPTION_SHOW_BYTE_OFFSET

    try:
        result_code = int(
            lib.spvBinaryToText(
                context,
                words,
                len(data) // 4,
                options,
                text_out,
                diagnostic_out,
            )
        )
        text = ""
        if text_out[0] != ffi.NULL:
            text = bytes(
                ffi.buffer(text_out[0].str, int(text_out[0].length))
            ).decode("utf-8", errors="replace")
        return NativeDisassemblyResult(
            ok=result_code == int(lib.SPV_SUCCESS),
            text=text,
            result_code=result_code,
            diagnostics=_diagnostics_from_out(ffi, diagnostic_out),
        )
    finally:
        if text_out[0] != ffi.NULL:
            lib.spvTextDestroy(text_out[0])
        if diagnostic_out[0] != ffi.NULL:
            lib.spvDiagnosticDestroy(diagnostic_out[0])
        lib.spvContextDestroy(context)


def validate_bytes(
    data: bytes, *, target_env: str = "universal_1_6"
) -> NativeValidationResult:
    if len(data) % 4 != 0:
        return NativeValidationResult(
            ok=False,
            result_code=-4,
            diagnostics=(_non_word_aligned_diagnostic(),),
        )

    ffi, lib = _ffi_and_lib()
    context = _create_context(ffi, lib, target_env)

    options = lib.spvValidatorOptionsCreate()
    if options == ffi.NULL:
        lib.spvContextDestroy(context)
        raise RuntimeError("spvValidatorOptionsCreate failed")

    diagnostic_out = ffi.new("spv_diagnostic*")

    try:
        lib.spvValidatorOptionsSetFriendlyNames(options, True)
        words = ffi.from_buffer("const uint32_t[]", data)
        binary = ffi.new("spv_const_binary_t*", (words, len(data) // 4))

        result_code = int(
            lib.spvValidateWithOptions(
                context,
                options,
                binary,
                diagnostic_out,
            )
        )

        return NativeValidationResult(
            ok=result_code == int(lib.SPV_SUCCESS),
            result_code=result_code,
            diagnostics=_diagnostics_from_out(ffi, diagnostic_out),
        )
    finally:
        if diagnostic_out[0] != ffi.NULL:
            lib.spvDiagnosticDestroy(diagnostic_out[0])
        lib.spvValidatorOptionsDestroy(options)
        lib.spvContextDestroy(context)
