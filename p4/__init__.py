import copy
import difflib
import logging
import re
import subprocess
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Callable, Iterator, NewType, Protocol, TypedDict, override

import requests
from ast_grep_py import SgRoot
from joblib import Parallel, delayed
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool  # pyright: ignore[reportUnknownVariableType]
from langgraph.prebuilt import (
    create_react_agent,  # pyright: ignore[reportUnknownVariableType]
)
from pydantic import BaseModel, field_serializer
from ripgrepy import Ripgrepy  # pyright: ignore[reportMissingTypeStubs]

from p4_core.environment.protocols import BaseEnvironment
from p4_core.pattern.models import Fragment
from p4_core.pattern.protocols import BasePattern
from p4_core.policy.models import BaseMessage, Prompt, SystemMessage
from p4_core.policy.protocols import BaseChatPolicy
from p4_core.runnable.protocols import BaseRunnable
from p4_core.scope.protocols import Scope

Symbol = NewType("Symbol", str)


class BaseDocument(BaseModel):
    value: str

    def annotated(
        self, patterns: list[BasePattern], opening_tag: str, closing_tag: str
    ):
        fragments: list[Fragment] = list()
        current_end = 0
        for fragment in sorted(
            [
                fragment
                for pattern in patterns
                for fragment in pattern.match(self.value)
            ],
            key=lambda x: x.start_position,
        ):
            if fragment.start_position >= current_end:
                fragments.append(fragment)
                current_end = fragment.end_position

        value = self.value

        offset = 0

        for fragment in fragments:
            value = (
                value[: fragment.start_position + offset]
                + opening_tag
                + fragment.value
                + closing_tag
                + value[fragment.end_position + offset :]
            )

            offset += len(opening_tag) + len(closing_tag)

        while f"{opening_tag}{opening_tag}" in value:
            value = value.replace(
                f"{opening_tag}{opening_tag}",
                opening_tag,
            )
        while f"{closing_tag}{closing_tag}" in value:
            value = value.replace(
                f"{closing_tag}{closing_tag}",
                closing_tag,
            )

        annotation = copy.deepcopy(self)
        annotation.value = value

        return annotation

    def __hash__(self) -> int:
        return hash(self.value)

    def as_markdown(self) -> str:
        raise NotImplementedError("as_markdown() must be implemented in subclasses")


class TextDocument(BaseDocument):
    def __hash__(self) -> int:
        return hash(self.value)

    def as_markdown(self) -> str:
        return f"## (Uneditable)\n\n```\n{self.value.strip()}\n```"


class FileDocument(BaseDocument):
    relative_path: Path
    source_directory: Path

    @field_serializer("relative_path")
    def _serialize_relative_path(self, value: Path) -> str:
        return str(value)

    @field_serializer("source_directory")
    def _serialize_source_directory(self, value: Path) -> str:
        return str(value)

    def __hash__(self) -> int:
        return hash(f"{self.relative_path}:{self.value}")

    def as_markdown(self) -> str:
        return f"### {self.relative_path}\n\n```\n{self.value.strip()}\n```"


Document = TextDocument | FileDocument


class GlobalCommandContext(TypedDict):
    source_directory: Path
    global_executable: Path


def _symbol_locations(
    name: str,
    context: GlobalCommandContext,
):
    stdout = subprocess.check_output(
        [
            context["global_executable"],
            "-x",
            name,
        ],
        cwd=context["source_directory"],
    ).decode("utf-8", errors="ignore")

    for group in re.finditer(r"(\w+)\s+(\d+)\s+([\w\/.-]+)\s+", stdout):
        row = int(group.group(2)) - 1
        relative_path = Path(group.group(3))

        # FIXME: Hardcoded for now
        if (
            "aflplusplus" in relative_path.parts
            or "libfuzzer" in relative_path.parts
            or "fuzztest" in relative_path.parts
            or "fuzz" in relative_path.parts
            or "libfuzzer" in relative_path.parts
        ):
            continue

        yield (relative_path, row)


class BaseTool(BaseRunnable[Symbol, set[Document], GlobalCommandContext]): ...


class CppFunctionDefinitionTool(BaseTool):
    def run(self, x: Symbol, context: GlobalCommandContext) -> set[Document]:
        result: set[Document] = set()
        maybe_function_name = (
            x.split(".")[-1].split("(")[0].split("<")[0].split("::")[-1]
        )
        for relative_path, row in _symbol_locations(maybe_function_name, context):
            text = (context["source_directory"] / relative_path).read_text(
                encoding="utf-8", errors="ignore"
            )

            root = SgRoot(text, "cpp").root()

            for node in root.find_all(kind="function_definition"):
                if node.range().start.line <= row <= node.range().end.line:
                    result.add(
                        FileDocument(
                            value=node.text(),
                            relative_path=relative_path,
                            source_directory=context["source_directory"],
                        )
                    )

        return result


class CppTypeDefinitionTool(BaseTool):
    def run(self, x: Symbol, context: GlobalCommandContext) -> set[Document]:
        result: set[Document] = set()
        maybe_type_name = x.split(".")[-1].split("(")[0].split("<")[0].split("::")[-1]
        for relative_path, row in _symbol_locations(maybe_type_name, context):
            text = (context["source_directory"] / relative_path).read_text(
                encoding="utf-8", errors="ignore"
            )
            root = SgRoot(text, "cpp").root()
            for node in root.find_all(kind="type_definition"):
                if node.range().start.line <= row <= node.range().end.line:
                    result.add(
                        FileDocument(
                            value=node.text(),
                            relative_path=relative_path,
                            source_directory=context["source_directory"],
                        )
                    )

        return result


class JavaMethodDeclarationTool(BaseTool):
    def run(self, x: Symbol, context: GlobalCommandContext) -> set[Document]:
        result: set[Document] = set()
        maybe_method_name = x.split(".")[-1].split("(")[0]
        for relative_path, row in _symbol_locations(maybe_method_name, context):
            text = (context["source_directory"] / relative_path).read_text(
                encoding="utf-8", errors="ignore"
            )
            root = SgRoot(text, "java").root()
            for node in root.find_all(kind="method_declaration"):
                if node.range().start.line <= row <= node.range().end.line:
                    result.add(
                        FileDocument(
                            value=node.text(),
                            relative_path=relative_path,
                            source_directory=context["source_directory"],
                        )
                    )

        return result


class AIxCCEnvironment[Context](BaseEnvironment[set[Document], set[Symbol], Context]):
    def __init__(
        self,
        tools: list[BaseTool],
        episode_length: int,
        scope_builder: Callable[[Context], Scope],
    ):
        if len(tools) == 0:
            raise ValueError("At least one tool must be provided.")

        self._tools = tools
        self._episode_length = episode_length
        self._current_step = 0
        self._scope_builder = scope_builder

    def reset(self, context: Context) -> set[Document]:
        self._current_step = 0
        initial_crash_log = self._scope_builder(context)["initial_crash_log"]

        maybe_stack_trace = re.findall(
            r"^    #\d+ 0x[0-9a-f]+ in .*$", initial_crash_log, re.MULTILINE
        )
        if len(maybe_stack_trace) > 0:
            return {
                TextDocument(value="\n".join(maybe_stack_trace)),
            }
        else:
            return {
                TextDocument(value=initial_crash_log),
            }

    @override
    def step(
        self,
        action: set[Symbol],
        observation: set[TextDocument | FileDocument],
        context: Context,
    ):
        self._current_step += 1
        result = super().step(action, observation, context)
        return result

    def _step(
        self, action: set[Symbol], observation: set[Document], context: Context
    ) -> tuple[set[Document], bool, bool]:
        scope = self._scope_builder(context)
        documents: set[Document] = {
            document
            for documents in Parallel(n_jobs=-1, backend="threading")(
                delayed(tool.run_or_none)(symbol, scope)
                for symbol, tool in product(action, self._tools)
            )
            if documents is not None
            for document in documents
        } | observation

        terminated = self._current_step >= self._episode_length

        return documents, terminated, False


class CppCallPattern(BasePattern):
    def __init__(self, limit: int | None):
        self._limit = limit

    def match(self, source: str) -> set[Fragment]:
        root = SgRoot(source, "cpp").root()
        return {
            Fragment(value=node.text(), start_position=node.range().start.index)
            for node in root.find_all(kind="call_expression")[: self._limit]
        }


class CppTypeIdentifierPattern(BasePattern):
    def __init__(self, limit: int | None):
        self._limit = limit

    def match(self, source: str) -> set[Fragment]:
        root = SgRoot(source, "cpp").root()
        return {
            Fragment(
                value=node.text(),
                start_position=node.range().start.index,
            )
            for node in root.find_all(kind="type_identifier")[: self._limit]
        }


class JavaInvocationPattern(BasePattern):
    def __init__(self, limit: int | None):
        self._limit = limit

    def match(self, source: str) -> set[Fragment]:
        root = SgRoot(source, "java").root()
        return {
            Fragment(
                value=node.text(),
                start_position=node.range().start.index,
            )
            for node in root.find_all(kind="method_invocation")[: self._limit]
        }


class JazzerFunctionSignaturePattern(BasePattern):
    def __init__(self, limit: int | None):
        self._limit = limit

    def match(self, source: str) -> set[Fragment]:
        if "	at " not in source:
            return set()

        return {
            Fragment(
                value=matched.group(1),
                start_position=matched.start(1),
            )
            for matched in list(re.finditer(r"\tat (.*)\(", source))[: self._limit]
        }


class SanitizerFunctionSignaturePattern(BasePattern):
    def __init__(self, limit: int | None):
        self._limit = limit

    def match(self, source: str) -> set[Fragment]:
        if "    #" not in source:
            return set()

        return {
            Fragment(
                value=matched.group(1),
                start_position=matched.start(1),
            )
            for matched in list(
                re.finditer(r"    #\d+ 0x[a-f0-9]+ in (.+) \/", source)
            )[: self._limit]
        }


class BaseEraserPolicy(BaseChatPolicy[set[Document], set[Symbol]]):
    def __init__(
        self,
        patterns: list[BasePattern],
        opening_tag: str = "<cc>",
        closing_tag: str = "</cc>",
        system_message: SystemMessage = {
            "role": "system",
            "content": """Format:
## Reasoning
Explain the potential vulnerabilities and provide a brief reasoning for the selection of relevant symbols.
## Relevant Symbols
1. `symbol`
2. `symbol`
3. `symbol`

Detailed rules:

1. **## Reasoning**

   * Explain the reasoning behind the selection of relevant symbols.
   * Ignore and **do not list** anything related to fuzzers (e.g. afl++, libFuzzer, honggfuzz).

2. **## Relevant Symbols**

   * Rank the **relevant** symbols based on your reasoning bullets.
   * List them in descending order of relevance, numbered, each enclosed in back-ticks.
   * Provide **no explanations** in this section—just the rank and the symbol.
""",
        },
    ):
        if len(patterns) == 0:
            raise ValueError("At least one pattern must be provided.")

        self._patterns = patterns
        self._opening_tag = opening_tag
        self._closing_tag = closing_tag
        self._system_message = system_message

    def prompt_from_observation(
        self,
        observation: set[Document],
        previous_observation: set[Document],
    ) -> Prompt:
        annotations = {
            document.annotated(self._patterns, self._opening_tag, self._closing_tag)
            for document in observation
        }

        return [
            {
                "role": "user",
                "content": "# Instruction\n\n"
                + self._system_message["content"]
                + "---\n\n"
                "# Contents\n\n"
                + "\n\n".join(document.as_markdown() for document in annotations),
            },
        ]

    def action_from_completion(
        self, completion: BaseMessage, prompt: Prompt
    ) -> set[Symbol]:
        assert prompt[-1]["role"] == "user"

        values = [
            matched.group(1).lstrip(self._opening_tag).rstrip(self._closing_tag)
            for matched in re.finditer(
                r"\`([^\`]+)\`",
                completion["content"].split("## Relevant Symbols")[-1],
            )
        ]

        return {
            Symbol(value)
            for value in values
            if value != ""
            and (
                f"{self._opening_tag}{value}{self._closing_tag}"
                in prompt[-1]["content"]
            )
        }


class _VirtualFileSystem:
    def __init__(self, source_directory: Path):
        self._source_directory = source_directory
        self._overlay: dict[Path, str] = {}

    def write_text(self, path: Path, data: str):
        relative_path = self.resolve(path)
        self._overlay[relative_path] = data

    def read_text(self, path: Path):
        relative_path = self.resolve(path)
        if relative_path in self._overlay:
            return self._overlay[relative_path]
        else:
            return (self._source_directory / relative_path).read_text(
                encoding="utf-8", errors="ignore"
            )

    def glob(self, pattern: str) -> list[Path]:
        return list(self._source_directory.glob(pattern))

    def grep(self, relative_path: Path, pattern: str) -> str:
        relative_path = self.resolve(relative_path)

        return (
            Ripgrepy(
                pattern,
                str(self._source_directory / relative_path),
            )
            .with_filename()
            .line_number()
            .context(16)
            .run()
            .as_string
        )

    def resolve(self, path: Path) -> Path:
        if path.is_absolute():
            if path.is_relative_to(self._source_directory):
                relative_path = path.relative_to(self._source_directory)
            else:
                raise ValueError(
                    "Path must be in the source directory."
                    f" Given path: {path}, source directory: {self._source_directory}"
                )
        else:
            if path.is_relative_to(self._source_directory):
                relative_path = path.relative_to(self._source_directory)
            else:
                relative_path = path

        if not (self._source_directory / relative_path).exists():
            raise FileNotFoundError(
                f"Path {path} does not exist in source directory {self._source_directory}."
            )

        return relative_path

    @property
    def diff(self) -> str | None:
        diff = ""

        for path, content in self._overlay.items():
            assert not path.is_absolute(), "Path must be relative."

            before = (self._source_directory / path).read_text(
                encoding="utf-8", errors="ignore"
            )

            after = content
            diff_of_file = difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
            diff_of_file = "".join(diff_of_file)
            if diff_of_file:
                diff += f"{diff_of_file}\n"

        match diff:
            case "":
                return None
            case diff:
                return diff

    @property
    def relative_patches(self) -> dict[Path, str]:
        assert all(not path.is_absolute() for path in self._overlay.keys())
        return {path: content for path, content in self._overlay.items()}


def generate_patch_using_langchain(
    documents: set[Document], source_directory: Path, chat_model: BaseChatModel
):
    file_system = _VirtualFileSystem(source_directory)

    @tool
    def edit(
        file_path: str,
        search: str,
        replace: str,
    ):
        """Replace the content of a file with the provided content.
        Explain the reasoning before the change in the `explain` tool.

        Args:
            file_path: The path to the file to be modified.
            search: A exact match of the text to be replaced.
            replace: The new text to replace the old one.
        """
        relative_path = file_system.resolve(Path(file_path))

        before = file_system.read_text(relative_path)
        if search not in before:
            raise ValueError(f"Search text not found in file {relative_path}.")
        if before.count(search) > 1:
            raise ValueError(
                f"Search text '{search}' found multiple times in file {relative_path}. Please ensure it is unique."
            )

        after = before.replace(search, replace)
        file_system.write_text(relative_path, after)

        return f"File {relative_path} modified successfully. The change has been applied to the target file."

    @tool
    def explain(reasoning: str):
        """Explain the reasoning behind identified vulnerabilities and fixes before making changes.

        Args:
            reasoning: A detailed explanation of which change is needed, which vulnerability it fixes, and which part of the file it addresses. Identify the vulnerability, the affected code region, the exploit scenario, and why this fix cures it.
        """
        return "User understands the reasoning."

    agent = create_react_agent(model=chat_model, tools=[edit, explain])

    agent.invoke(
        {
            "messages": [
                HumanMessage(
                    """Role: Security Expert with Full Discretionary Authority
Mission: Using `edit` tool, fix the vulnerabilities in the provided code files based on the crash log and the code fragments.

Guiding Principles:
- Base all findings and fixes strictly on the actual content of the file.
- Prioritize true security fixes over general code hygiene.
- Evaluate inter-file and inter-function interactions—modern software is interconnected and must be audited as such.

Strict Prohibitions:
- NEVER fix or modify files outside the source directory.
- NEVER modify fuzzing harnesses.
- NEVER introduce or rely on fuzzer-specific macros or flags.
- NEVER delete or comment out any functionality unless it clearly constitutes a backdoor.
- NEVER remove assert() or abort() statements guarded by fuzz-specific flags like FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION.\n"""
                    "Crash log and possibly vulnerable code fragments to be fixed:\n"
                    + "\n\n".join(document.as_markdown() for document in documents)
                )
            ],
        },
        {
            "recursion_limit": 128,
        },
    )

    return file_system.relative_patches, file_system.diff


class BaseClient[T](Protocol):
    _vllm_base_url: str
    _adapter_base_url: str

    def _as(self, id: str) -> T: ...

    @contextmanager
    def enabled(
        self,
        id: str,
        crash_log: str,
        patterns: list[BasePattern],
        tools: list[BaseTool],
        context: GlobalCommandContext,
    ) -> Iterator[T | None]:
        fragments = {
            fragment for pattern in patterns for fragment in pattern.match(crash_log)
        }
        symbols = {Symbol(fragment.value) for fragment in fragments}
        documents = {
            document
            for documents_or_none in [
                tool.run_or_none(symbol, context)
                for symbol in symbols
                for tool in tools
            ]
            if documents_or_none is not None
            for document in documents_or_none
        }
        text = "\n".join(document.value for document in documents)

        try:
            adapter_path = self.adapt(id=id, text=text)
        except Exception as e:
            logging.exception(e)
            yield None
            return

        try:
            self.load_lora_adapter(
                lora_name=id,
                lora_path=adapter_path,
            )
        except Exception as e:
            logging.exception(e)
            yield None
            return

        try:
            yield self._as(id=id)
        finally:
            try:
                self.unload_lora_adapter(lora_name=id)
            except Exception as e:
                logging.exception(e)

    def load_lora_adapter(self, lora_name: str, lora_path: Path):
        response = requests.post(
            f"{self._vllm_base_url}/load_lora_adapter",
            json={
                "lora_name": lora_name,
                "lora_path": str(lora_path),
            },
        )
        response.raise_for_status()
        return response

    def unload_lora_adapter(self, lora_name: str):
        response = requests.post(
            f"{self._vllm_base_url}/unload_lora_adapter",
            json={
                "lora_name": lora_name,
            },
        )
        response.raise_for_status()
        return response

    def adapt(
        self,
        id: str,
        text: str,
        block_size: int = 256,
        learning_rate: float = 1e-8,
        per_device_train_batch_size: int = 8,
        num_train_epochs: int = 64,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ) -> Path:
        response = requests.post(
            f"{self._adapter_base_url}/adapt",
            json={
                "id": id,
                "text": text,
                "block_size": block_size,
                "learning_rate": learning_rate,
                "per_device_train_batch_size": per_device_train_batch_size,
                "num_train_epochs": num_train_epochs,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
            },
        )
        response.raise_for_status()

        return Path(response.json()["lora_path"])
