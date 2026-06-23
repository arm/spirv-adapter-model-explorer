#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
from datetime import datetime, timezone
from typing import List

from spdx_tools.common.spdx_licensing import spdx_licensing
from spdx_tools.spdx.model import (
    Actor,
    ActorType,
    CreationInfo,
    Document,
    ExtractedLicensingInfo,
    Package,
    PackagePurpose,
    Relationship,
    RelationshipType,
)
from spdx_tools.spdx.validation.document_validator import (
    validate_full_spdx_document,
)
from spdx_tools.spdx.validation.validation_message import ValidationMessage
from spdx_tools.spdx.writer.write_anything import write_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate SPDX document for SPIRV-Tools binary builds"
    )
    parser.add_argument(
        "--spirv-tools-ref", required=True, help="SPIRV-Tools reference"
    )
    parser.add_argument(
        "--spirv-headers-ref", required=True, help="SPIRV-Headers reference"
    )
    parser.add_argument(
        "--wheel-version",
        required=True,
        help="Version of the spirv-adapter-model-explorer wheel",
    )
    parser.add_argument(
        "--output",
        default="spirv-tools-builds.spdx.json",
        help="Output file path",
    )

    args = parser.parse_args()

    build_time = datetime.now(timezone.utc)
    build_time_str = build_time.strftime("%Y%m%d-%H%M%S")
    binary_version = (
        f"{args.wheel_version}-spirv_tools-{args.spirv_tools_ref}"
        f"-spirv_headers-{args.spirv_headers_ref}-{build_time_str}"
    )

    creation_info = CreationInfo(
        spdx_version="SPDX-2.3",
        spdx_id="SPDXRef-DOCUMENT",
        name="spirv-adapter-model-explorer-spirv-tools-builds",
        data_license="CC0-1.0",
        document_namespace=(
            "https://arm.com/spdx/"
            "spirv-adapter-model-explorer-spirv-tools-"
            f"{args.spirv_tools_ref}-spirv_headers-"
            f"{args.spirv_headers_ref}-{build_time_str}"
        ),
        creators=[
            Actor(
                ActorType.ORGANIZATION,
                "Arm Limited",
                "open-source-office@arm.com",
            )
        ],
        creator_comment='THIS SOFTWARE BILL OF MATERIALS ("SBOM") IS PROVIDED BY ARM LIMITED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT ARE DISCLAIMED. IN NO EVENT SHALL ARM LIMITED BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SBOM, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.',
        created=build_time,
    )
    document = Document(creation_info)

    spirv_tools_pkg = Package(
        name="SPIRV-Tools",
        spdx_id="SPDXRef-SPIRVTools",
        download_location="https://github.com/KhronosGroup/SPIRV-Tools",
        version=args.spirv_tools_ref,
        files_analyzed=False,
        license_declared=spdx_licensing.parse("Apache-2.0"),
        primary_package_purpose=PackagePurpose.LIBRARY,
        description="Base SPIRV-Tools package used for the packaged native library build.",
        originator=Actor(ActorType.ORGANIZATION, "Khronos Group"),
    )

    spirv_headers_pkg = Package(
        name="SPIRV-Headers",
        spdx_id="SPDXRef-SPIRVHeaders",
        download_location="https://github.com/KhronosGroup/SPIRV-Headers",
        version=args.spirv_headers_ref,
        files_analyzed=False,
        license_declared=spdx_licensing.parse(
            "LicenseRef-Khronos-SPIRV-Headers AND MIT AND CC-BY-4.0"
        ),
        primary_package_purpose=PackagePurpose.LIBRARY,
        description="SPIR-V headers package used when building SPIRV-Tools.",
        originator=Actor(ActorType.ORGANIZATION, "Khronos Group"),
    )

    spirv_tools_bin = Package(
        name="SPIRV-Tools-shared",
        spdx_id="SPDXRef-SPIRVToolsSharedBin",
        download_location=(
            "https://pypi.org/project/spirv-adapter-model-explorer/"
        ),
        version=binary_version,
        files_analyzed=False,
        license_declared=spdx_licensing.parse("Apache-2.0"),
        primary_package_purpose=PackagePurpose.LIBRARY,
        description="Binary build of SPIRV-Tools shared library, distributed as part of spirv-adapter-model-explorer Python wheel.",
        originator=Actor(
            ActorType.ORGANIZATION, "Arm Limited", "open-source-office@arm.com"
        ),
    )

    document.packages = [
        spirv_tools_pkg,
        spirv_headers_pkg,
        spirv_tools_bin,
    ]
    document.extracted_licensing_info = [
        ExtractedLicensingInfo(
            license_id="LicenseRef-Khronos-SPIRV-Headers",
            license_name="Khronos SPIR-V Headers License",
            extracted_text=(
                "The Khronos SPIR-V Headers license applies to files in the "
                "SPIRV-Headers repository except files with separate license "
                "callouts. See the upstream SPIRV-Headers LICENSE file for "
                "the complete license text."
            ),
            cross_references=[
                "https://github.com/KhronosGroup/SPIRV-Headers/blob/main/LICENSE"
            ],
        )
    ]

    relationships = [
        Relationship(
            "SPDXRef-DOCUMENT",
            RelationshipType.DESCRIBES,
            "SPDXRef-SPIRVToolsSharedBin",
        ),
        Relationship(
            "SPDXRef-SPIRVToolsSharedBin",
            RelationshipType.DEPENDS_ON,
            "SPDXRef-SPIRVTools",
        ),
        Relationship(
            "SPDXRef-SPIRVToolsSharedBin",
            RelationshipType.DEPENDS_ON,
            "SPDXRef-SPIRVHeaders",
        ),
        Relationship(
            "SPDXRef-SPIRVTools",
            RelationshipType.DEPENDS_ON,
            "SPDXRef-SPIRVHeaders",
        ),
    ]
    document.relationships = relationships

    validation_messages: List[ValidationMessage] = validate_full_spdx_document(
        document
    )
    for m in validation_messages:
        logging.warning(m.validation_message)
        logging.warning(m.context)

    assert not validation_messages, (
        "SPDX document validation failed. See above for details."
    )

    write_file(document, args.output)
    print(f"SPDX document generated: {args.output}")


if __name__ == "__main__":
    main()
