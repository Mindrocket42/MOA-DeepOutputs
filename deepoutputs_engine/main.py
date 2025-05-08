import time
import os
import logging
from pathlib import Path
from deepoutputs_engine.config import logger
from deepoutputs_engine.utils import read_prompt_from_file, sanitize_filename
from deepoutputs_engine.agents.mixture import MixtureOfAgents
from deepoutputs_engine.tracing import Tracer
from deepoutputs_engine.reports import generate_markdown_report, generate_detailed_markdown_report

async def main():
    try:
        # Setup run ID and tracer
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        prompt = read_prompt_from_file()
        base_filename = sanitize_filename(prompt)
        run_id = f"{base_filename}_{timestamp}"
        tracer = Tracer(run_id=run_id, enabled=True)

        # Import config
        from deepoutputs_engine.config import (
            AGENT1_MODEL, AGENT2_MODEL, AGENT3_MODEL,
            DEEP_RESEARCH_AGENT_MODEL,
            MOA_NUM_LAYERS, INCLUDE_DEEP_RESEARCH,
            OUTPUT_DIR
        )

        # Use INCLUDE_DEEP_RESEARCH from config
        logger.info(f"INCLUDE_DEEP_RESEARCH from config: {INCLUDE_DEEP_RESEARCH}")

        # Create run-specific directory
        run_dir = Path(OUTPUT_DIR) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Set up file logging for this run
        log_file = run_dir / f"{run_id}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Build models list
        models_list = [AGENT1_MODEL, AGENT2_MODEL, AGENT3_MODEL]
        if INCLUDE_DEEP_RESEARCH:
            logger.info(f"Including DeepResearch agent with model: {DEEP_RESEARCH_AGENT_MODEL}")
            models_list.append(DEEP_RESEARCH_AGENT_MODEL)
        else:
            logger.info("DeepResearch agent not included in this run.")

        logger.info(f"Starting MOA workflow with run_id: {run_id}")
        logger.info(f"Using models: {models_list}")

        # Log initial configuration
        tracer.log_workflow_event("Configuration", {
            "models": models_list,
            "num_layers": MOA_NUM_LAYERS,
            "include_deep_research": INCLUDE_DEEP_RESEARCH,
            "prompt": prompt
        })

        moa = MixtureOfAgents(
            models=models_list,
            num_layers=MOA_NUM_LAYERS,
            include_deep_research=INCLUDE_DEEP_RESEARCH,
            tracer=tracer
        )

        # --- Modular Orchestration Logic ---
        logger.info("Starting workflow execution")
        tracer.log_workflow_event("Workflow Start", {"timestamp": time.time()})
        
        workflow_results = await moa.run_workflow(prompt)
        final_output = workflow_results["final_output"]
        layer_outputs = workflow_results["layer_outputs"]
        
        # Log workflow completion
        tracer.log_workflow_event("Workflow Complete", {
            "timestamp": time.time(),
            "num_layers": len(layer_outputs),
            "final_output_length": len(final_output)
        })
        
        logger.info("Workflow execution completed successfully")

        # Report generation
        prompt_for_report = workflow_results["prompt"]
        layer_details = workflow_results["layer_outputs"]
        final_response = workflow_results["final_output"]

        # Patch: Ensure each layer dict has a 'layer_number' key for report compatibility
        for layer in layer_details:
            if "layer" in layer:
                layer["layer_number"] = layer["layer"]

        # Calculate utilization: similarity between each agent's output and the final output
        from difflib import SequenceMatcher
        utilization = {}
        
        # Get initial responses from the first layer
        if layer_details and isinstance(layer_details, list) and "initial_responses" in layer_details[0]:
            initial_responses = layer_details[0]["initial_responses"]
            for agent_name, output in initial_responses:
                similarity = SequenceMatcher(None, output, final_response).ratio() * 100
                utilization[agent_name] = similarity

        # Log utilization metrics
        tracer.log_workflow_event("Agent Utilization", {
            "metrics": utilization,
            "final_agent": moa.final_agent.name
        })

        final_agent_name = moa.final_agent.name

        # Generate reports
        logger.info("Generating reports")
        markdown_report = generate_markdown_report(
            prompt_for_report,
            layer_details,
            final_response,
            utilization,
            final_agent_name,
            moa
        )
        detailed_report = generate_detailed_markdown_report(
            prompt_for_report,
            layer_details,
            final_response,
            utilization,
            final_agent_name,
            getattr(moa, "synthesis_agent", None).name if getattr(moa, "synthesis_agent", None) else "",
            getattr(moa, "devils_advocate_agent", None).name if getattr(moa, "devils_advocate_agent", None) else "",
            moa
        )

        # Save reports to run-specific directory
        # Save markdown report
        markdown_path = run_dir / "report.md"
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_report)
        logger.info(f"Saved markdown report to: {markdown_path}")

        # Save detailed report
        detailed_path = run_dir / "detailed_report.md"
        with open(detailed_path, "w", encoding="utf-8") as f:
            f.write(detailed_report)
        logger.info(f"Saved detailed report to: {detailed_path}")

        # Log report generation
        tracer.log_workflow_event("Reports Generated", {
            "markdown_report_path": str(markdown_path),
            "detailed_report_path": str(detailed_path)
        })

        logger.info("Workflow completed successfully")
        logger.info(f"All outputs saved to: {run_dir}")

    except Exception as e:
        logger.error(f"Error in main workflow: {str(e)}", exc_info=True)
        if 'tracer' in locals():
            tracer.log_workflow_event("Error", {
                "error": str(e),
                "traceback": str(e.__traceback__)
            }, error=str(e))
        raise
    finally:
        # Cleanup
        if 'tracer' in locals():
            tracer.close()
        if 'moa' in locals():
            await moa.close_client()
        # Remove file handler
        if 'file_handler' in locals():
            logger.removeHandler(file_handler)
            file_handler.close()

# Entry point
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())