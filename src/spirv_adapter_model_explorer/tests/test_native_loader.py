# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from spirv_adapter_model_explorer.native import loader


def test_platform_dir_supports_windows_arm64(monkeypatch):
    monkeypatch.setattr(loader.platform, "system", lambda: "Windows")
    monkeypatch.setattr(loader.platform, "machine", lambda: "ARM64")

    assert loader._platform_dir() == "win_arm64"
