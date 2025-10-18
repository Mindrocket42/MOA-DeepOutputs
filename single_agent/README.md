# Single Agent Workflow Engine

A sophisticated single-agent workflow system that uses specialized cognitive functions to provide high-quality, well-reasoned responses. Built on the principles of Mixture of Agents (MOA) but optimized for single-model execution with cognitive specialization.

## Overview

The Single Agent Workflow Engine implements a cognitive architecture where different specialized models handle different thinking functions:

- **Response Agent**: Initial analysis and comprehensive reasoning
- **Devils Advocate**: Critical challenge and contrarian viewpoints
- **Synthesis Agent**: Integration with dissent management
- **Final Agent**: Authoritative decision making

## Key Features

- **Cognitive Function Specialization**: Each model optimized for specific reasoning tasks
- **Layered Processing**: Configurable depth for complex problem solving
- **Dissent Management**: Critical viewpoints preserved and surfaced throughout
- **Self-Review Mechanisms**: Built-in quality assurance and validation
- **Comprehensive Tracing**: Full observability and performance metrics
- **Flexible Configuration**: Environment-based configuration system

## Installation

The Single Agent Engine is part of the MOA-DeepOutputs project. Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Configure the single agent models in your `.env` file:

```env
# Single Agent Models
SA_RESPONSE_AGENT_MODEL="anthropic/claude-3.5-sonnet"
SA_DEVILS_ADVOCATE_AGENT_MODEL="openai/gpt-4o"
SA_SYNTHESIS_AGENT_MODEL="google/gemini-2.0-flash-001"
SA_FINAL_AGENT_MODEL="anthropic/claude-3.5-haiku"

# Workflow Configuration
SA_NUM_LAYERS=2
SA_ENABLE_SELF_REVIEW=true
SA_OUTPUT_DIR="single_agent/reports"
SA_TRACE_DIR="single_agent/traces"

# Token Limits
SA_RESPONSE_MAX_TOKENS=8000
SA_REVIEW_MAX_TOKENS=4000
SA_DEVILS_ADVOCATE_MAX_TOKENS=6000
SA_SYNTHESIS_MAX_TOKENS=8000
SA_FINAL_MAX_TOKENS=6000
```

## Usage

### Command Line Interface

Run the single agent workflow from the command line:

```bash
# Run with default prompt.txt
python -m single_agent.single_agent_engine.main

# Specify custom prompt file
python -m single_agent.single_agent_engine.main --prompt-file my_prompt.txt

# Run in dry-run mode
python -m single_agent.single_agent_engine.main --dry-run

# Enable verbose logging
python -m single_agent.single_agent_engine.main --verbose
```

### Programmatic Usage

Use the single agent workflow in your Python code:

```python
import asyncio
from single_agent.single_agent_engine import run_single_agent_workflow

async def main():
    prompt = "What are the best practices for software testing?"

    result = await run_single_agent_workflow(
        prompt=prompt,
        enable_tracing=True
    )

    print("Final Answer:", result['final_output'])
    print("Processing Time:", result['execution_time'], "seconds")

# Run the workflow
asyncio.run(main())
```

### Advanced Usage with Custom Configuration

```python
import asyncio
from single_agent.single_agent_engine import SingleAgentWorkflow, SingleAgentTracer

async def custom_workflow():
    # Initialize with custom tracer
    tracer = SingleAgentTracer(run_id="custom_analysis", enabled=True)
    workflow = SingleAgentWorkflow(tracer=tracer)

    prompt = "Analyze the impact of artificial intelligence on employment."

    try:
        result = await workflow.run_workflow(prompt)

        print("Workflow completed successfully!")
        print(f"Layers processed: {result['metadata']['num_layers']}")
        print(f"Total tokens: {result['metadata']['total_tokens']}")

        return result

    finally:
        await workflow.close_client()
        if tracer:
            tracer.close()

asyncio.run(custom_workflow())
```

## Workflow Architecture

### Processing Pipeline

```
User Prompt
    ↓
Response Agent (Initial Analysis)
    ↓
Self-Review (Optional Quality Check)
    ↓
Devils Advocate (Critical Challenge)
    ↓
Synthesis Agent (Integration & Dissent Management)
    ↓
[Repeat for N Layers]
    ↓
Final Agent (Authoritative Decision)
    ↓
Comprehensive Report
```

### Cognitive Functions

#### Response Agent
- **Purpose**: Provide comprehensive initial analysis
- **Focus**: Logic, evidence, practical insights
- **Output**: Well-reasoned response with assumptions tagged

#### Devils Advocate
- **Purpose**: Stress-test and challenge assumptions
- **Focus**: Critical analysis, edge cases, alternative viewpoints
- **Output**: Specific critiques and contrarian solutions

#### Synthesis Agent
- **Purpose**: Integrate insights while preserving dissent
- **Focus**: Balanced analysis with dissenting viewpoints clearly marked
- **Output**: Unified narrative with critique sections

#### Final Agent
- **Purpose**: Authoritative decision making
- **Focus**: Evidence-based conclusions with uncertainty assessment
- **Output**: Final answer with justification and limitations

## Output Structure

The workflow generates comprehensive outputs:

```
single_agent/
├── reports/
│   ├── report.md              # Standard summary report
│   └── detailed_report.md     # Full analysis with prompts
├── traces/
│   └── json/
│       └── [run_id]/
│           └── trace_[timestamp].jsonl  # Detailed execution trace
└── dry_runs/                  # Dry-run outputs (if enabled)
```

## Configuration Options

### Workflow Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `SA_NUM_LAYERS` | Number of processing layers | 2 |
| `SA_ENABLE_SELF_REVIEW` | Enable self-review phase | true |
| `SA_RESPONSE_MAX_TOKENS` | Max tokens for response phase | 8000 |
| `SA_DEVILS_ADVOCATE_MAX_TOKENS` | Max tokens for devils advocate | 6000 |
| `SA_SYNTHESIS_MAX_TOKENS` | Max tokens for synthesis | 8000 |
| `SA_FINAL_MAX_TOKENS` | Max tokens for final answer | 6000 |

### Model Configuration

Each cognitive function can use a different model, allowing for specialization:

- **Response Agent**: Analytical models good at comprehensive reasoning
- **Devils Advocate**: Critical thinking models good at finding flaws
- **Synthesis Agent**: Integrative models good at balancing perspectives
- **Final Agent**: Authoritative models good at final decision making

## Quality Assurance Features

### Dissent Management
- Critical viewpoints are preserved throughout the workflow
- Dissenting points are clearly marked in synthesis phases
- Final answers reference and address critiques

### Self-Review Mechanisms
- Optional self-assessment phase for response validation
- Confidence level tagging throughout
- Assumption identification and validation

### Layered Validation
- Multi-layer processing for complex problems
- Progressive refinement of analysis
- Convergence validation across layers

## Performance Considerations

### Token Optimization
- Configurable token limits per phase
- Efficient prompt construction
- Token usage tracking and reporting

### Concurrency Control
- Shared HTTP client with connection pooling
- Configurable concurrency limits
- Automatic retry with exponential backoff

### Caching and Optimization
- Response caching for repeated operations
- Efficient context management
- Performance metrics collection

## Troubleshooting

### Common Issues

#### Import Errors
```
ModuleNotFoundError: No module named 'single_agent_engine'
```
**Solution**: Ensure you're running from the project root directory and the module is properly installed.

#### API Connection Issues
```
httpx.HTTPStatusError: 429 Too Many Requests
```
**Solution**: Check your API rate limits and adjust `OPENROUTER_CONCURRENCY` in your `.env` file.

#### Configuration Errors
```
ValueError: OPENROUTER_API_KEY environment variable is required
```
**Solution**: Ensure your `.env` file contains the required API keys.

### Debugging

Enable verbose logging for detailed execution information:

```bash
python -m single_agent.single_agent_engine.main --verbose
```

Check trace files for detailed execution analysis:

```bash
# View recent traces
ls single_agent/traces/json/
cat single_agent/traces/json/[run_id]/trace_*.jsonl | jq .
```

## Examples

### Simple Analysis
```python
from single_agent.single_agent_engine import run_single_agent_workflow
import asyncio

result = asyncio.run(run_single_agent_workflow(
    "What are the ethical implications of artificial intelligence?"
))
print(result['final_output'])
```

### Complex Multi-Layer Analysis
```python
from single_agent.single_agent_engine import SingleAgentWorkflow
import asyncio

async def deep_analysis():
    workflow = SingleAgentWorkflow()
    result = await workflow.run_workflow(
        "Analyze the potential economic impacts of quantum computing on cryptography."
    )
    return result

result = asyncio.run(deep_analysis())
print(f"Analysis completed in {result['execution_time']:.2f}s")
```

## Integration with MOA

The Single Agent Workflow is designed to complement the existing MOA system:

- **Shared Infrastructure**: Uses same API clients and configuration patterns
- **Compatible Outputs**: Generates reports in similar formats
- **Unified Configuration**: Extends existing `.env` configuration
- **Performance Tracking**: Integrates with existing metrics systems

## Contributing

When contributing to the Single Agent Workflow:

1. Maintain cognitive function separation
2. Preserve dissent management mechanisms
3. Update tests for new functionality
4. Follow existing code patterns and documentation standards

## License

This project is part of MOA-DeepOutputs and follows the same license terms.