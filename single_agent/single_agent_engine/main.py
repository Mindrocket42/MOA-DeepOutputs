"""
Main entry point for Single Agent Workflow.

Provides command-line interface and programmatic access to run
single agent workflows with comprehensive reporting and tracing.
"""

import asyncio
import time
import os
import logging
from pathlib import Path
from typing import Optional

from .config import logger, SA_OUTPUT_DIR, SA_TRACE_DIR
from .utils import read_prompt_from_file, sanitize_filename, ensure_directory, format_timestamp_for_filename
from .workflow import SingleAgentWorkflow
from .tracing import SingleAgentTracer
from .reports import generate_markdown_report, generate_detailed_markdown_report, generate_workflow_summary

async def main(
    prompt_file: str = "prompt.txt",
    output_dir: Optional[str] = None,
    enable_tracing: bool = True,
    dry_run: bool = False
) -> dict:
    """
    Main entry point for single agent workflow execution.

    Args:
        prompt_file: Path to file containing the prompt
        output_dir: Custom output directory (overrides config)
        enable_tracing: Whether to enable detailed tracing
        dry_run: Whether this is a dry run (affects output location)

    Returns:
        Dictionary with workflow results and metadata
    """
    try:
        # Setup run configuration
        timestamp = format_timestamp_for_filename()
        prompt = read_prompt_from_file(prompt_file)
        base_filename = sanitize_filename(prompt)
        run_id = f"{base_filename}_{timestamp}"

        # Determine output directories
        if dry_run:
            from .config import SA_DRY_RUN_DIR
            actual_output_dir = SA_DRY_RUN_DIR
        else:
            actual_output_dir = output_dir or SA_OUTPUT_DIR

        reports_dir = ensure_directory(actual_output_dir)
        traces_dir = ensure_directory(SA_TRACE_DIR)

        # Setup logging for this run
        log_file = reports_dir / f"{run_id}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Initialize tracer
        tracer = SingleAgentTracer(run_id=run_id, enabled=enable_tracing) if enable_tracing else None

        logger.info(f"Starting Single Agent Workflow with run_id: {run_id}")
        logger.info(f"Prompt file: {prompt_file}")
        logger.info(f"Output directory: {reports_dir}")
        logger.info(f"Tracing enabled: {enable_tracing}")
        logger.info(f"Dry run: {dry_run}")

        if tracer:
            tracer.log_workflow_event("Configuration", {
                "prompt_file": prompt_file,
                "output_dir": str(reports_dir),
                "tracing_enabled": enable_tracing,
                "dry_run": dry_run,
                "run_id": run_id
            })

        # Initialize and run workflow
        workflow = SingleAgentWorkflow(tracer=tracer)

        logger.info("Executing Single Agent Workflow...")
        start_time = time.time()

        workflow_results = await workflow.run_workflow(prompt)

        execution_time = time.time() - start_time
        logger.info(f"Workflow execution completed in {execution_time:.2f}s")

        # Extract results
        final_output = workflow_results["final_output"]
        layer_outputs = workflow_results["layer_outputs"]
        metadata = workflow_results["metadata"]

        # Generate reports
        logger.info("Generating reports...")

        # Standard markdown report
        markdown_report = generate_markdown_report(
            prompt, layer_outputs, final_output, metadata, workflow
        )

        # Detailed markdown report
        detailed_report = generate_detailed_markdown_report(
            prompt, layer_outputs, final_output, metadata, workflow, tracer
        )

        # Save reports
        markdown_path = reports_dir / "report.md"
        detailed_path = reports_dir / "detailed_report.md"

        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_report)

        with open(detailed_path, "w", encoding="utf-8") as f:
            f.write(detailed_report)

        logger.info(f"Saved standard report to: {markdown_path}")
        logger.info(f"Saved detailed report to: {detailed_path}")

        # Generate workflow summary
        summary = generate_workflow_summary(metadata, tracer)
        logger.info(f"Workflow Summary:\n{summary}")

        # Log completion
        if tracer:
            tracer.log_workflow_event("Reports Generated", {
                "markdown_report_path": str(markdown_path),
                "detailed_report_path": str(detailed_path),
                "execution_time": execution_time
            })

        logger.info("Single Agent Workflow completed successfully!")
        logger.info(f"All outputs saved to: {reports_dir}")

        # Prepare return results
        results = {
            "run_id": run_id,
            "prompt": prompt,
            "final_output": final_output,
            "layer_outputs": layer_outputs,
            "metadata": metadata,
            "reports": {
                "markdown": str(markdown_path),
                "detailed": str(detailed_path)
            },
            "execution_time": execution_time,
            "success": True
        }

        return results

    except Exception as e:
        logger.error(f"Error in Single Agent Workflow: {str(e)}", exc_info=True)

        error_results = {
            "run_id": locals().get("run_id", "unknown"),
            "error": str(e),
            "success": False,
            "execution_time": time.time() - locals().get("start_time", time.time())
        }

        if 'tracer' in locals() and tracer:
            tracer.log_workflow_event("Error", {
                "error": str(e),
                "traceback": str(e.__traceback__)
            })

        raise

    finally:
        # Cleanup
        if 'tracer' in locals() and tracer:
            tracer.close()
        if 'workflow' in locals():
            await workflow.close_client()
        # Remove file handler
        if 'file_handler' in locals():
            logger.removeHandler(file_handler)
            file_handler.close()

async def run_single_agent_workflow(
    prompt: str,
    output_dir: Optional[str] = None,
    enable_tracing: bool = True
) -> dict:
    """
    Programmatic interface to run a single agent workflow with a string prompt.

    Args:
        prompt: The prompt string to process
        output_dir: Custom output directory
        enable_tracing: Whether to enable tracing

    Returns:
        Dictionary with workflow results
    """
    # Create temporary prompt file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(prompt)
        temp_prompt_file = f.name

    try:
        return await main(
            prompt_file=temp_prompt_file,
            output_dir=output_dir,
            enable_tracing=enable_tracing,
            dry_run=False
        )
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_prompt_file)
        except:
            pass

# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Single Agent Workflow Engine")
    parser.add_argument(
        "--prompt-file", "-p",
        default="prompt.txt",
        help="Path to prompt file (default: prompt.txt)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for reports (overrides config)"
    )
    parser.add_argument(
        "--no-tracing",
        action="store_true",
        help="Disable detailed tracing"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (outputs to dry_run directory)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Run the workflow
    try:
        results = asyncio.run(main(
            prompt_file=args.prompt_file,
            output_dir=args.output_dir,
            enable_tracing=not args.no_tracing,
            dry_run=args.dry_run
        ))

        print(f"\n✅ Single Agent Workflow completed successfully!")
        print(f"📁 Outputs saved to: {os.path.dirname(results['reports']['markdown'])}")
        print(f"📄 Run ID: {results['run_id']}")
        print(".2f")

    except KeyboardInterrupt:
        print("\n⚠️  Workflow interrupted by user")
    except Exception as e:
        print(f"\n❌ Workflow failed: {str(e)}")
        exit(1)