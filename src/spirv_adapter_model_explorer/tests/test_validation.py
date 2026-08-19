# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from spirv_adapter_model_explorer.native.loader import (
    SpirvToolsUnavailableError,
    spirv_tools_library_path,
)
from spirv_adapter_model_explorer.validation import validate_spirv

from .conftest import FIXTURES_ROOT
from .test_spirv_parser import assemble_valid_spirv


def require_vendored_spirv_tools() -> None:
    try:
        spirv_tools_library_path()
    except SpirvToolsUnavailableError as err:
        pytest.skip(str(err))


INVALID_FIXTURES = {"semantic-model.spvasm"}


@pytest.mark.parametrize(
    "fixture",
    [
        fixture
        for fixture in sorted(FIXTURES_ROOT.glob("*.spvasm"))
        if fixture.name not in INVALID_FIXTURES
    ],
    ids=lambda fixture: fixture.name,
)
def test_spvasm_fixture_is_valid(fixture):
    require_vendored_spirv_tools()

    assemble_valid_spirv(fixture.read_text(encoding="utf-8"))


def test_validate_known_good_spirv():
    require_vendored_spirv_tools()

    binary = assemble_valid_spirv(
        """
; SPIR-V
; Version: 1.0
; Bound: 6
; Schema: 0
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %3 "main"
OpExecutionMode %3 LocalSize 1 1 1
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
OpReturn
OpFunctionEnd
""",
    )

    result = validate_spirv(binary)

    assert result.ok
    assert result.diagnostics == ()


def test_validate_rejects_bad_magic():
    require_vendored_spirv_tools()

    binary = bytearray(
        assemble_valid_spirv(
            """
; SPIR-V
; Version: 1.0
; Bound: 6
; Schema: 0
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %3 "main"
OpExecutionMode %3 LocalSize 1 1 1
%1 = OpTypeVoid
%2 = OpTypeFunction %1
%3 = OpFunction %1 None %2
%4 = OpLabel
OpReturn
OpFunctionEnd
""",
        )
    )
    binary[:4] = b"BAD!"

    result = validate_spirv(bytes(binary))

    assert not result.ok
    assert result.diagnostics
    assert result.diagnostics[0].severity == "error"
    assert result.diagnostics[0].message


def test_validate_rejects_non_word_aligned_binary():
    result = validate_spirv(b"abc")

    assert not result.ok
    assert result.diagnostics
    assert result.diagnostics[0].position is None
