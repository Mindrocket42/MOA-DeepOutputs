import time
from deepoutputs_engine.config import logger
from deepoutputs_engine.utils import read_prompt_from_file, sanitize_filename
from deepoutputs_engine.agents.mixture import MixtureOfAgents
from deepoutputs_engine.tracing import Tracer
from deepoutputs_engine.reports import generate_markdown_report, generate_detailed_markdown_report

async def main():
    # Setup run ID and tracer
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prompt = read_prompt_from_file()
    base_filename = sanitize_filename(prompt)
    run_id = f"{base_filename}_{timestamp}"
    tracer = Tracer(run_id=run_id, enabled=True)

    # Import config
    from deepoutputs_engine.config import (
        AGENT1_MODEL, AGENT2_MODEL, AGENT3_MODEL,
        MOA_NUM_LAYERS, INCLUDE_DEEP_RESEARCH
    )

    models_list = [AGENT1_MODEL, AGENT2_MODEL, AGENT3_MODEL]
    moa = MixtureOfAgents(
        models=models_list,
        num_layers=MOA_NUM_LAYERS,
        include_deep_research=INCLUDE_DEEP_RESEARCH,
        tracer=tracer
    )

    # --- Modular Orchestration Logic ---
    workflow_results = await moa.run_workflow(prompt)
    final_output = workflow_results["final_output"]
    layer_outputs = workflow_results["layer_outputs"]

    # Report generation (can be expanded to use layer_outputs for detailed reports)
    markdown_report = generate_markdown_report(final_output)
    detailed_report = generate_detailed_markdown_report(final_output)

    # TODO: Save reports to file system as per plan

    # Cleanup
    tracer.close()
    await moa.close_client()

# Entry point
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())