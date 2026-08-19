# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch, sentinel

import pytest

from spirv_adapter_model_explorer import validation
from spirv_adapter_model_explorer.main import (
    ADAPTER_CONFIG_EDITORS_AVAILABLE,
    SHOW_GRAPH_CONSTANTS_SETTING,
    SpirvAdapter,
)
from spirv_adapter_model_explorer.native.loader import (
    SpirvToolsUnavailableError,
)
from spirv_adapter_model_explorer.validation import (
    SpirvValidationDiagnostic,
    SpirvValidationResult,
)

from .test_builder_spv_integration import MULTI_INPUT_OUTPUT_GRAPH_ASM
from .test_spirv_parser import assemble_valid_spirv


def _write_words(path, words: list[int]) -> None:
    path.write_bytes(b"".join(word.to_bytes(4, "little") for word in words))


def _write_valid_graph_spirv(path) -> None:
    path.write_bytes(assemble_valid_spirv(MULTI_INPUT_OUTPUT_GRAPH_ASM))


def _validation_failure(message: str) -> SpirvValidationResult:
    return SpirvValidationResult(
        ok=False,
        result_code=1,
        diagnostics=(
            SpirvValidationDiagnostic(
                severity="error",
                message=message,
                position=5,
                line=1,
                column=2,
            ),
        ),
    )


def test_adapter_metadata_supports_spirv_assembly_files():
    assert "spvasm" in SpirvAdapter.metadata.fileExts


def test_adapter_config_editors_match_model_explorer_capability():
    result = SpirvAdapter().get_config_editors()

    assert result.error == ""
    assert result.configEditors is not None
    if not ADAPTER_CONFIG_EDITORS_AVAILABLE:
        assert result.configEditors == []
        return

    assert len(result.configEditors) == 1
    editor = result.configEditors[0]
    assert editor.id == SHOW_GRAPH_CONSTANTS_SETTING
    assert editor.label == "Show graph constants"
    assert editor.defaultValue is False


def test_convert_hides_graph_constant_nodes_by_default(tmp_path):
    spv_path = tmp_path / "graph.spv"
    spv_path.write_bytes(b"SPIR-V")
    with patch(
        "spirv_adapter_model_explorer.main.build_spirv_graphs_from_bytes",
        return_value=[sentinel.graph],
    ) as mock_build:
        result = SpirvAdapter().convert(str(spv_path), {})

    mock_build.assert_called_once_with(b"SPIR-V", show_graph_constants=False)
    assert result == {"graphs": [sentinel.graph]}


def test_convert_shows_graph_constant_nodes_when_enabled(tmp_path):
    spv_path = tmp_path / "graph.spv"
    spv_path.write_bytes(b"SPIR-V")
    with patch(
        "spirv_adapter_model_explorer.main.build_spirv_graphs_from_bytes",
        return_value=[sentinel.graph],
    ) as mock_build:
        result = SpirvAdapter().convert(
            str(spv_path), {SHOW_GRAPH_CONSTANTS_SETTING: True}
        )

    mock_build.assert_called_once_with(b"SPIR-V", show_graph_constants=True)
    assert result == {"graphs": [sentinel.graph]}


def test_spvasm_file_is_assembled_before_conversion(tmp_path):
    spvasm_path = tmp_path / "graph.spvasm"
    spvasm_path.write_text(MULTI_INPUT_OUTPUT_GRAPH_ASM, encoding="utf-8")

    result = SpirvAdapter().convert(str(spvasm_path), {})

    graph = result["graphs"][0]
    assert graph.id.startswith("graph_")
    assert [node.label for node in graph.nodes] == [
        "Input 0",
        "Input 1",
        "ADD",
        "ADD",
        "Output 0",
        "Output 1",
    ]
    assert graph.groupNodeAttributes[""]["Validation"] == "Passed"


def test_invalid_spvasm_reports_assembly_diagnostics(tmp_path):
    spvasm_path = tmp_path / "bad.spvasm"
    spvasm_path.write_text("OpDefinitelyNotAnInstruction\n", encoding="utf-8")

    with pytest.raises(ValueError) as err:
        SpirvAdapter().convert(str(spvasm_path), {})

    assert "Failed to assemble SPIR-V text" in str(err.value)
    assert "OpDefinitelyNotAnInstruction" in str(err.value)


def test_validation_pass_is_attached_to_successfully_parsed_graph(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        validation,
        "validate_spirv",
        lambda _data: SpirvValidationResult(ok=True, diagnostics=()),
    )
    spv_path = tmp_path / "valid.spv"
    _write_valid_graph_spirv(spv_path)

    result = SpirvAdapter().convert(str(spv_path), {})

    graph = result["graphs"][0]
    assert graph.groupNodeAttributes is not None
    graph_attrs = graph.groupNodeAttributes[""]
    assert graph_attrs["Validation"] == "Passed"
    assert "spv full asm" in graph_attrs


def test_validation_attrs_are_inserted_before_existing_graph_attrs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        validation,
        "validate_spirv",
        lambda _data: SpirvValidationResult(ok=True, diagnostics=()),
    )
    spv_path = tmp_path / "valid_graph.spv"
    spv_path.write_bytes(assemble_valid_spirv(MULTI_INPUT_OUTPUT_GRAPH_ASM))

    result = SpirvAdapter().convert(str(spv_path), {})

    graph_attrs = result["graphs"][0].groupNodeAttributes[""]
    assert list(graph_attrs)[:3] == [
        "Validation",
        "graph type",
        "entry_points",
    ]


def test_validation_unavailable_is_attached_to_successfully_parsed_graph(
    tmp_path, monkeypatch
):
    def unavailable(_data):
        raise SpirvToolsUnavailableError("validator unavailable")

    monkeypatch.setattr(validation, "validate_spirv", unavailable)
    spv_path = tmp_path / "validation_unavailable.spv"
    _write_valid_graph_spirv(spv_path)

    result = SpirvAdapter().convert(str(spv_path), {})

    graph = result["graphs"][0]
    assert graph.groupNodeAttributes is not None
    graph_attrs = graph.groupNodeAttributes[""]
    assert graph_attrs["Validation"] == "Unavailable"
    assert graph_attrs["Validation diagnostics"] == (
        "SPIRV-Tools validator is unavailable."
    )


def test_validation_failure_is_attached_to_successfully_parsed_graph(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        validation,
        "validate_spirv",
        lambda _data: _validation_failure("validation says no"),
    )
    spv_path = tmp_path / "invalid_but_parseable.spv"
    _write_valid_graph_spirv(spv_path)

    result = SpirvAdapter().convert(str(spv_path), {})

    graph = result["graphs"][0]
    assert graph.id.startswith("graph_")
    assert [node.label for node in graph.nodes] == [
        "Input 0",
        "Input 1",
        "ADD",
        "ADD",
        "Output 0",
        "Output 1",
    ]
    assert graph.groupNodeAttributes is not None
    graph_attrs = graph.groupNodeAttributes[""]
    assert graph_attrs["Validation"] == "Failed"
    assert graph_attrs["Validation diagnostics"] == (
        "error: validation says no (word 5, line 1, column 2)"
    )


def test_validation_failure_is_included_when_binary_parsing_also_fails(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        validation,
        "validate_spirv",
        lambda _data: _validation_failure("bad magic from validator"),
    )
    spv_path = tmp_path / "invalid_and_unparseable.spv"
    _write_words(spv_path, [0x21444142, 0x00010000, 0, 1, 0])

    with pytest.raises(ValueError) as err:
        SpirvAdapter().convert(str(spv_path), {})

    message = str(err.value)
    assert "Failed to parse SPIR-V binary" in message
    assert "Invalid SPIR-V magic: got 0x21444142" in message
    assert "SPIR-V validation failed before parsing" in message
    assert (
        "error: bad magic from validator (word 5, line 1, column 2)" in message
    )
