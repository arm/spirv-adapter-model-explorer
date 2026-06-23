# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import platform
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = REPO_ROOT / ".native" / "build"
PACKAGE_VENDOR_ROOT = (
    REPO_ROOT / "src" / "spirv_adapter_model_explorer" / "vendor"
)


def platform_dir() -> str:
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Windows" and machine in {"amd64", "x86_64"}:
        return "win_amd64"
    if system == "Linux" and machine in {"amd64", "x86_64"}:
        return "linux_x86_64"
    if system == "Linux" and machine in {"aarch64", "arm64"}:
        return "linux_aarch64"
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return "macos_arm64"
    if system == "Darwin" and machine in {"amd64", "x86_64"}:
        return "macos_x86_64"

    raise RuntimeError(f"Unsupported platform: {system} {machine}")


def library_name() -> str:
    system = platform.system()
    if system == "Windows":
        return "SPIRV-Tools-shared.dll"
    if system == "Darwin":
        return "libSPIRV-Tools-shared.dylib"
    return "libSPIRV-Tools-shared.so"


def find_built_library() -> Path:
    name = library_name()
    matches = sorted(path for path in BUILD_ROOT.rglob(name) if path.is_file())
    if not matches:
        raise FileNotFoundError(
            f"Could not find {name} under {BUILD_ROOT}. Run "
            "`hatch run build-native` first."
        )

    # Prefer the SPIRV-Tools source build output over any staged/test copies
    # that might also exist under the native build directory.
    for path in matches:
        if "spirv-tools-build" in path.parts and "source" in path.parts:
            return path

    return matches[0]


def main() -> None:
    source = find_built_library()
    target_dir = PACKAGE_VENDOR_ROOT / platform_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / library_name()

    shutil.copy2(source, target)
    print(f"Staged {source} -> {target}")


if __name__ == "__main__":
    main()
