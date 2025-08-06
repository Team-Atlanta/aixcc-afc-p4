# P4: Policy-Pattern-Program-Patch Library

P4 is an common library for automated vulnerability analysis and patching with custom model powered agent. It combines LLMs with reinforcement learning to analyze crash logs, identify vulnerable code patterns, and generate targeted patches.

## Key Features

- **Modular Architecture**: Composable components for pattern detection, policy execution, and environment management
- **Multi-Language Support**: C++ and Java analysis with extensible pattern matching
- **LLM Integration**: Policy-driven decision making using language models
- **LoRA Adaptation**: Dynamic model fine-tuning based on vulnerability context
- **Type-Safe Design**: Protocol-based interfaces with generic type parameters

## Architecture

### Core Components

#### Pattern Layer (`p4_core.pattern`)

- **`BasePattern`**: Protocol for pattern matching
- **`Fragment`**: Code segments with position metadata
- Language-specific patterns: `CppCallPattern`, `JavaInvocationPattern`, etc.

#### Policy Layer (`p4_core.policy`)

- **`BaseChatPolicy`**: LLM-driven decision making
- **`BaseEraserPolicy`**: Vulnerability symbol extraction
- Three-phase execution: Observation → Prompt → Completion → Action

#### Environment Layer (`p4_core.environment`)

- **`BaseEnvironment`**: Step-based execution context
- **`BaseTrainableEnvironmentWithChat`**: RL support with reward functions

#### Runnable Layer (`p4_core.runnable`)

- **`BaseRunnable`**: Generic execution protocol with error handling

#### Scope Layer (`p4_core.scope`)

- **`Scope`**: Execution context management
- **`BaseSandbox`**: Isolated execution environments

### Main Framework (`p4`)

#### Document Management

- **`BaseDocument`**: Abstract base for all document types
- **`TextDocument`**: Uneditable text content (crash logs)
- **`FileDocument`**: Source code files with path metadata
- **Annotation System**: Pattern-based highlighting

#### Tools

- **`CppFunctionDefinitionTool`**: AST-based C++ function extraction
- **`JavaMethodDeclarationTool`**: Java method identification
- **Symbol Resolution**: Integration with `global` command-line tool

#### AI Environment (`AIxCCEnvironment`)

- Multi-tool parallel execution using `joblib`
- Episode-based execution with configurable limits
- Automatic crash log parsing and stack trace extraction

#### Patch Generation

- **LangChain Integration**: Automated patch generation
- **Virtual File System**: In-memory file modifications
- **Tool-based Editing**: Structured search-and-replace operations

#### Model Adaptation

- **LoRA Client**: Dynamic adapter loading/unloading
- **Context-Aware Training**: Automatic adapter generation from vulnerability context

## Implementation

### Type Safety

```python
class BaseRunnable[T, U, Context](Protocol):
    def run(self, x: T, context: Context) -> U: ...
```

### Pattern Matching

Uses `ast-grep` for AST-based analysis:

```python
def match(self, source: str) -> set[Fragment]:
    root = SgRoot(source, "cpp").root()
    return {Fragment(value=node.text(), start_position=node.range().start.index)
            for node in root.find_all(kind="call_expression")}
```

### Parallel Execution

```python
documents = Parallel(n_jobs=-1, backend="threading")(
    delayed(tool.run_or_none)(symbol, scope)
    for symbol, tool in product(action, self._tools)
)
```

## Dependencies

- **`pydantic` (≥2.8.2)**: Type-safe data modeling
- **`langchain-core`**: LLM integration and tool orchestration
- **`langgraph` (≥0.4.7)**: Graph-based agent execution
- **`openai` (≥1.66.3)**: OpenAI API integration
- **`ast-grep-py` (≥0.38.3)**: AST-based code analysis
- **`ripgrepy` (≥2.1.0)**: High-performance text search
- **`joblib` (≥1.5.1)**: Parallel computing
- **`requests` (≥2.32.3)**: HTTP client

## Usage

### Basic Setup

```python
from p4 import AIxCCEnvironment, CppCallPattern, CppFunctionDefinitionTool

# Configure environment
patterns = [CppCallPattern(limit=10)]
tools = [CppFunctionDefinitionTool()]
environment = AIxCCEnvironment(tools=tools, episode_length=5, scope_builder=scope_builder)

# Execute analysis
observation = environment.reset(context)
action = policy.act(observation, previous_observation)
observation, terminated, truncated = environment.step(action, observation, context)
```

### Custom Pattern

```python
class CustomPattern(BasePattern):
    def match(self, source: str) -> set[Fragment]:
        root = SgRoot(source, "cpp").root()
        return {Fragment(value=node.text(), start_position=node.range().start.index)
                for node in root.find_all(pattern="<tree-sitter-kind>")}
```
