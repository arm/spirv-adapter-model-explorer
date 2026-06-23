<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->

# SPIR-V Adapter for Model Explorer

Standalone Model Explorer adapter for SPIR-V binary files.

Initial scope:

- parse `.spv` / `.spirv` binary modules with a pure-Python parser;
- use vendored SPIR-V grammar JSON resources for opcode/operand decoding;
- build reusable bytes-level APIs suitable for later VGF adapter integration;
- later add a small native validation extension using the public SPIRV-Tools C API.

VGF parsing is deliberately out of scope for this repo. The VGF adapter should extract embedded SPIR-V bytes and call this package's reusable APIs.

## Development

Prerequisites:

- Python 3.10 to 3.13
- CMake and a C/C++ build toolchain for the native SPIRV-Tools test support
- [Hatch](https://hatch.pypa.io/) for the standard development environment

Install Hatch if needed:

```sh
python -m pip install hatch
```

Run the full test workflow:

```sh
hatch run test
```

This configures and builds the native SPIRV-Tools dependency, stages it for tests, then runs `pytest -q`.

For quick pure-Python test iterations after setup:

```sh
python -m pytest -q
```

Run linting with the project dev environment:

```sh
hatch run ruff check src
```
