# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from model_explorer import Adapter, AdapterMetadata, ModelExplorerGraphs

try:
    from model_explorer import (
        AdapterGetConfigEditorsResult,  # pyright: ignore[reportAttributeAccessIssue]
    )
    from model_explorer.config_editor import SlideToggleConfigEditor
except ImportError:

    @dataclass
    class AdapterGetConfigEditorsResult:  # type: ignore[no-redef]
        """Fallback response for Model Explorer versions without adapter settings."""

        configEditors: list | None = None
        error: str = ""

    SlideToggleConfigEditor = None  # type: ignore[assignment]

from .assembly import SpirvAssemblyResult, assemble_spirv_text
from .builder import build_spirv_graphs_from_bytes
from .validation import format_validation_diagnostic

SHOW_GRAPH_CONSTANTS_SETTING = "show_graph_constants"
ADAPTER_CONFIG_EDITORS_AVAILABLE = SlideToggleConfigEditor is not None


class SpirvAdapter(Adapter):
    metadata = AdapterMetadata(
        id="spirv_adapter_model_explorer",
        name="SPIR-V Adapter",
        description="SPIR-V adapter for Model Explorer",
        fileExts=["spv", "spirv", "spvasm"],
    )

    def __init__(self):
        super().__init__()

    def get_config_editors(self) -> AdapterGetConfigEditorsResult:
        """Expose settings when the Model Explorer adapter API supports them."""
        if SlideToggleConfigEditor is None:
            return AdapterGetConfigEditorsResult(configEditors=[])
        return AdapterGetConfigEditorsResult(
            configEditors=[
                SlideToggleConfigEditor(
                    id=SHOW_GRAPH_CONSTANTS_SETTING,
                    label="Show graph constants",
                    defaultValue=False,
                    help="Show OpGraphConstantARM values as graph nodes.",
                )
            ]
        )

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        """Convert a SPIR-V binary or assembly file to Model Explorer graphs."""
        path = Path(model_path)
        data = _read_spirv_binary_or_assembly(path)
        graphs = build_spirv_graphs_from_bytes(
            data,
            show_graph_constants=bool(
                settings.get(SHOW_GRAPH_CONSTANTS_SETTING, False)
            ),
        )
        return {"graphs": graphs}


def _read_spirv_binary_or_assembly(path: Path) -> bytes:
    if path.suffix.lower() != ".spvasm":
        return path.read_bytes()

    result = assemble_spirv_text(path.read_text(encoding="utf-8"))
    if result.ok:
        return result.data
    raise _assembly_error(result)


def _assembly_error(result: SpirvAssemblyResult) -> ValueError:
    if not result.diagnostics:
        return ValueError("Failed to assemble SPIR-V text.")
    diagnostics = "\n".join(
        format_validation_diagnostic(diagnostic)
        for diagnostic in result.diagnostics
    )
    return ValueError(f"Failed to assemble SPIR-V text:\n{diagnostics}")
