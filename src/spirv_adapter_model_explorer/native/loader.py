# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import platform
from importlib.resources import files


class SpirvToolsUnavailableError(RuntimeError):
    """Raised when the vendored SPIRV-Tools shared library is unavailable."""


def _platform_dir() -> str:
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Windows" and machine in {"amd64", "x86_64"}:
        return "win_amd64"
    if system == "Windows" and machine in {"arm64", "aarch64"}:
        return "win_arm64"
    if system == "Linux" and machine in {"amd64", "x86_64"}:
        return "linux_x86_64"
    if system == "Linux" and machine in {"aarch64", "arm64"}:
        return "linux_aarch64"
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return "macos_arm64"
    if system == "Darwin" and machine in {"amd64", "x86_64"}:
        return "macos_x86_64"

    raise SpirvToolsUnavailableError(
        f"Unsupported platform for vendored SPIRV-Tools: {system} {machine}"
    )


def _library_name() -> str:
    system = platform.system()
    if system == "Windows":
        return "SPIRV-Tools-shared.dll"
    if system == "Darwin":
        return "libSPIRV-Tools-shared.dylib"
    return "libSPIRV-Tools-shared.so"


def spirv_tools_library_path() -> str:
    path = (
        files("spirv_adapter_model_explorer.vendor")
        / _platform_dir()
        / _library_name()
    )

    if not path.is_file():
        raise SpirvToolsUnavailableError(
            "Vendored SPIRV-Tools shared library is missing. Reinstall "
            "spirv-adapter-model-explorer from a built wheel, or build it "
            "from source so CMake can fetch and build SPIRV-Tools."
        )

    return str(path)
