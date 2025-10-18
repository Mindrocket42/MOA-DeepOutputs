#!/usr/bin/env python3
"""
Example usage of the Single Agent Workflow Engine.

This script demonstrates various ways to use the single agent workflow
for different types of analysis and problem-solving tasks.
"""

import asyncio
import os
from pathlib import Path

async def simple_example():
    """Simple example with basic prompt."""
    print("=== Simple Example ===")

    from single_agent.single_agent_engine import run_single_agent_workflow

    prompt = "What are the most important factors to consider when choosing a programming language for a new project?"

    print(f"Prompt: {prompt}")
    print("Processing...")

    result = await run_single_agent_workflow(
        prompt=prompt,
        enable_tracing=True
    )

    print("\n[SUCCESS] Completed!")
    print(f"Execution time: {result['execution_time']:.2f} seconds")
    print(f"Layers processed: {result['metadata']['num_layers']}")
    print(f"Total tokens: ~{result['metadata']['total_tokens']}")

    print("\nFinal Answer:")
    print(result['final_output'])

    return result

async def complex_analysis_example():
    """Example with complex multi-layered analysis."""
    print("\n=== Complex Analysis Example ===")

    from single_agent.single_agent_engine import SingleAgentWorkflow, SingleAgentTracer

    prompt = """
    Analyze the potential impacts of artificial general intelligence (AGI) on global society.
    Consider economic, social, ethical, and technological dimensions.
    Provide specific recommendations for preparing society for AGI development.
    """

    print(f"Complex prompt: {prompt.strip()}")
    print("Processing with layered analysis...")

    # Custom workflow with tracer
    tracer = SingleAgentTracer(run_id="agi_analysis", enabled=True)
    workflow = SingleAgentWorkflow(tracer=tracer)

    try:
        result = await workflow.run_workflow(prompt)

        print("\n[SUCCESS] Complex analysis completed!")
        print(f"Execution time: {result['execution_time']:.2f} seconds")
        print(f"Layers processed: {result['metadata']['num_layers']}")
        print(f"Self-review enabled: {result['metadata']['self_review_enabled']}")

        # Show summary of layer processing
        print("\nLayer Summary:")
        for i, layer in enumerate(result['layer_outputs']):
            print(f"  Layer {i+1}: Response + Devils Advocate + Synthesis completed")

        print("\nFinal Answer (truncated):")
        print(result['final_output'][:500] + "...")

        return result

    finally:
        await workflow.close_client()
        tracer.close()

async def main():
    """Run basic examples."""
    print("Single Agent Workflow Engine - Basic Usage Examples")
    print("=" * 60)

    try:
        # Run simple example
        result = await simple_example()
        print(f"\n[SUCCESS] Simple example completed in {result['execution_time']:.2f}s")

        # Run complex example
        result = await complex_analysis_example()
        print(f"\n[SUCCESS] Complex example completed in {result['execution_time']:.2f}s")

    except Exception as e:
        print(f"\n[ERROR] Example failed: {e}")
        return

    print(f"\n{'='*60}")
    print("Examples completed successfully!")
    print("See README.md for more detailed usage instructions.")

if __name__ == "__main__":
    # Ensure we're in the right directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Examples interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Examples failed with error: {e}")
        raise