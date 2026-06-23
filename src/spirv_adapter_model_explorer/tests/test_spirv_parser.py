# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from spirv_adapter_model_explorer.assembly import assemble_spirv_text
from spirv_adapter_model_explorer.native.loader import (
    SpirvToolsUnavailableError,
)
from spirv_adapter_model_explorer.spirv.spirv_grammar import load_core_grammar
from spirv_adapter_model_explorer.spirv.spirv_parser import (
    SPIRV_MAGIC,
    SpirvHeader,
    SpirvInstruction,
    SpirvModule,
    SpirvOperand,
    parse_spirv,
)
from spirv_adapter_model_explorer.validation import (
    SpirvValidationResult,
    validate_spirv,
)


def _format_validation_result(result: SpirvValidationResult) -> str:
    if result.diagnostics:
        return "\n".join(
            f"{diagnostic.severity}: {diagnostic.message}"
            for diagnostic in result.diagnostics
        )
    return f"SPIR-V validation returned result code {result.result_code}."


def assemble_spirv(asm: str, *, target_env: str = "universal_1_0") -> bytes:
    try:
        result = assemble_spirv_text(asm, target_env=target_env)
    except SpirvToolsUnavailableError as err:
        pytest.fail(str(err))

    if not result.ok:
        diagnostics = "\n".join(
            diagnostic.message for diagnostic in result.diagnostics
        )
        pytest.fail(
            "SPIR-V assembly failed:\n"
            f"result code: {result.result_code}\n"
            f"diagnostics:\n{diagnostics}"
        )
    return result.data


def assemble_valid_spirv(
    asm: str, *, target_env: str = "universal_1_0"
) -> bytes:
    binary = assemble_spirv(asm, target_env=target_env)
    result = validate_spirv(binary)
    assert result.ok, _format_validation_result(result)
    return binary


def assemble_invalid_spirv(
    asm: str,
    *,
    expected_diagnostic: str | None = None,
    target_env: str = "universal_1_0",
) -> bytes:
    binary = assemble_spirv(asm, target_env=target_env)
    result = validate_spirv(binary)
    assert not result.ok, "SPIR-V validation unexpectedly passed"
    if expected_diagnostic is not None:
        diagnostics = "\n".join(
            diagnostic.message for diagnostic in result.diagnostics
        )
        assert expected_diagnostic in diagnostics
    return binary


def test_parse_core_spirv_module():
    binary = assemble_valid_spirv(
        """
; SPIR-V
; Version: 1.0
; Bound: 7
; Schema: 0
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %2 "entry"
OpExecutionMode %2 LocalSize 1 1 1
%1 = OpString "hello"
OpName %2 "entry"
%3 = OpTypeVoid
%4 = OpTypeFunction %3
%2 = OpFunction %3 None %4
%5 = OpLabel
OpReturn
OpFunctionEnd
""",
    )

    module = parse_spirv(binary, load_core_grammar())

    assert module == SpirvModule(
        header=SpirvHeader(
            magic=SPIRV_MAGIC,
            version=0x00010000,
            generator=module.header.generator,
            bound=6,
            schema=0,
        ),
        instructions=(
            SpirvInstruction(
                index=0,
                word_offset=5,
                word_count=2,
                opcode=17,
                opname="OpCapability",
                operands=(
                    SpirvOperand(
                        kind="Capability",
                        display="Shader(1)",
                        name="Capability",
                    ),
                ),
                raw_words=module.instructions[0].raw_words,
            ),
            SpirvInstruction(
                index=1,
                word_offset=7,
                word_count=3,
                opcode=14,
                opname="OpMemoryModel",
                operands=(
                    SpirvOperand(
                        kind="AddressingModel",
                        display="Logical(0)",
                    ),
                    SpirvOperand(
                        kind="MemoryModel",
                        display="GLSL450(1)",
                    ),
                ),
                raw_words=module.instructions[1].raw_words,
            ),
            SpirvInstruction(
                index=2,
                word_offset=10,
                word_count=5,
                opcode=15,
                opname="OpEntryPoint",
                operands=module.instructions[2].operands,
                raw_words=module.instructions[2].raw_words,
            ),
            SpirvInstruction(
                index=3,
                word_offset=15,
                word_count=6,
                opcode=16,
                opname="OpExecutionMode",
                operands=module.instructions[3].operands,
                raw_words=module.instructions[3].raw_words,
            ),
            SpirvInstruction(
                index=4,
                word_offset=21,
                word_count=4,
                opcode=7,
                opname="OpString",
                operands=(
                    SpirvOperand(kind="IdResult", display="%1"),
                    SpirvOperand(
                        kind="LiteralString", display='"hello"', name="String"
                    ),
                ),
                raw_words=module.instructions[4].raw_words,
                result_id=1,
            ),
            SpirvInstruction(
                index=5,
                word_offset=25,
                word_count=4,
                opcode=5,
                opname="OpName",
                operands=(
                    SpirvOperand(
                        kind="IdRef",
                        display="%2",
                        ref_id=2,
                        name="Target",
                    ),
                    SpirvOperand(
                        kind="LiteralString", display='"entry"', name="Name"
                    ),
                ),
                raw_words=module.instructions[5].raw_words,
            ),
            SpirvInstruction(
                index=6,
                word_offset=29,
                word_count=2,
                opcode=19,
                opname="OpTypeVoid",
                operands=(SpirvOperand(kind="IdResult", display="%3"),),
                raw_words=module.instructions[6].raw_words,
                result_id=3,
            ),
            SpirvInstruction(
                index=7,
                word_offset=31,
                word_count=3,
                opcode=33,
                opname="OpTypeFunction",
                operands=(
                    SpirvOperand(kind="IdResult", display="%4"),
                    SpirvOperand(
                        kind="IdRef",
                        display="%3",
                        ref_id=3,
                        name="Return Type",
                    ),
                ),
                raw_words=module.instructions[7].raw_words,
                result_id=4,
            ),
            SpirvInstruction(
                index=8,
                word_offset=34,
                word_count=5,
                opcode=54,
                opname="OpFunction",
                operands=(
                    SpirvOperand(
                        kind="IdResultType",
                        display="%3",
                    ),
                    SpirvOperand(kind="IdResult", display="%2"),
                    SpirvOperand(
                        kind="FunctionControl",
                        display="0x00000000",
                    ),
                    SpirvOperand(
                        kind="IdRef",
                        display="%4",
                        ref_id=4,
                        name="Function Type",
                    ),
                ),
                raw_words=module.instructions[8].raw_words,
                result_type_id=3,
                result_id=2,
            ),
            SpirvInstruction(
                index=9,
                word_offset=39,
                word_count=2,
                opcode=248,
                opname="OpLabel",
                operands=(SpirvOperand(kind="IdResult", display="%5"),),
                raw_words=module.instructions[9].raw_words,
                result_id=5,
            ),
            SpirvInstruction(
                index=10,
                word_offset=41,
                word_count=1,
                opcode=253,
                opname="OpReturn",
                operands=(),
                raw_words=module.instructions[10].raw_words,
            ),
            SpirvInstruction(
                index=11,
                word_offset=42,
                word_count=1,
                opcode=56,
                opname="OpFunctionEnd",
                operands=(),
                raw_words=module.instructions[11].raw_words,
            ),
        ),
    )


def test_parse_tosa_extinst_grammar():
    grammar = load_core_grammar()

    conv2d = grammar.extinst_instruction_spec("TOSA.001000.1", 2)

    assert conv2d is not None
    assert conv2d.opname == "CONV2D"
    assert tuple(operand.name for operand in conv2d.operands) == (
        "pad",
        "stride",
        "dilation",
        "acc_type",
        "local_bound",
        "input",
        "weight",
        "bias",
        "input_zp",
        "weight_zp",
    )


def test_parse_tosa_extinst_grammar_preserves_repeated_operands():
    grammar = load_core_grammar()

    concat = grammar.extinst_instruction_spec("TOSA.001000.1", 54)

    assert concat is not None
    assert concat.opname == "CONCAT"
    assert tuple(operand.name for operand in concat.operands) == (
        "axis",
        "input1",
    )
    assert tuple(operand.quantifier for operand in concat.operands) == (
        None,
        "*",
    )
