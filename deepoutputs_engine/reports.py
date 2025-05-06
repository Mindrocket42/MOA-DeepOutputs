from typing import List, Dict, Any
from deepoutputs_engine.utils import sanitize_for_markdown
from deepoutputs_engine.agents.mixture import MixtureOfAgents

def generate_markdown_report(
    prompt: str,
    layer_details: List[Dict[str, Any]],
    final_response: str,
    utilization: Dict[str, float],
    final_agent_name: str,
    moa: MixtureOfAgents
) -> str:
    """Generates the final summary Markdown report."""
    markdown_content = f"""# MoA Response Report

## Original Prompt
> {prompt}

## Agent Utilization
{chr(10).join([f"- {agent_name}: {percentage:.2f}%" for agent_name, percentage in utilization.items()])}

*(Note: Utilization is a heuristic based on text similarity to the final output)*

## Final MoA Response
**Final Response Agent:** {final_agent_name}

{sanitize_for_markdown(final_response)}
"""
    return markdown_content

def generate_detailed_markdown_report(
    prompt: str,
    layer_details: List[Dict[str, Any]],
    final_response: str,
    utilization: Dict[str, float],
    final_agent_name: str,
    synthesis_agent_name: str,
    devils_advocate_agent_name: str,
    moa: MixtureOfAgents
) -> str:
    """Generates a detailed comprehensive Markdown report with all intermediate outputs."""
    synthesis_model = getattr(moa.synthesis_agent, 'model', 'unknown')
    devils_advocate_model = getattr(moa.devils_advocate_agent, 'model', 'unknown')
    final_agent_model = getattr(moa.final_agent, 'model', 'unknown')

    markdown_content = f"""# MoA Detailed Response Report

## Original Prompt
> {prompt}

## Agent Utilization
{chr(10).join(
    [f"- {agent.name}: {utilization.get(agent.name, 0.0):.2f}%" for agent in moa.agents] +
    ([f"- Deep Research Agent: {utilization['Deep Research Agent']:.2f}%"] if 'Deep Research Agent' in utilization else []) +
    [f"- {agent_name}: {percentage:.2f}%" for agent_name, percentage in utilization.items()
     if agent_name not in [agent.name for agent in moa.agents] and agent_name != "Deep Research Agent"]
)}

*(Note: Utilization is a heuristic based on text similarity to the final output.{' Deep Research Agent utilization is included if present.' if 'Deep Research Agent' in utilization else ''})*

## Intermediate Outputs
"""

    for layer in layer_details:
        layer_num = layer["layer_number"]
        markdown_content += f"""
### Layer {layer_num}

#### Layer Prompt
> {layer["layer_prompt_details"]}

#### Step 1 - Agents Initial Responses
"""
        if layer.get("initial_responses") and isinstance(layer["initial_responses"], list):
            for item in layer["initial_responses"]:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    agent_name, response = item
                    agent_model = next((agent.model for agent in moa.agents if agent.name == agent_name), "unknown")
                    markdown_content += f"""\n##### {agent_name} - `{agent_model}`\n\n{sanitize_for_markdown(response)}\n"""
                else:
                    markdown_content += f"\n_Invalid format for initial response item: {item}_"
        else:
            markdown_content += "\n_No initial responses recorded or invalid format._"

        markdown_content += """
#### Step 2 - Agent Aggregation of All Responses
"""
        if layer.get("aggregation_responses") and isinstance(layer["aggregation_responses"], list):
            for item in layer["aggregation_responses"]:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    agent_name, response = item
                    agent_model = next((agent.model for agent in moa.agents if agent.name == agent_name), "unknown")
                    markdown_content += f"""\n##### {agent_name} - `{agent_model}`\n\n{sanitize_for_markdown(response)}\n"""
                else:
                    markdown_content += f"\n_Invalid format for aggregation response item: {item}_"
        else:
            markdown_content += "\n_No aggregation responses recorded or invalid format._"

        markdown_content += f"""\n#### Step 3 - Synthesized Aggregated Responses (Synthesis Agent: {synthesis_agent_name} - `{synthesis_model}`)

##### Synthesis

{sanitize_for_markdown(layer.get("synthesis", "N/A"))}

##### Devil's Advocate (Agent: {devils_advocate_agent_name} - `{devils_advocate_model}`)

{sanitize_for_markdown(layer.get("devils_advocate", "N/A"))}

---
"""

    markdown_content += f"""
## Information Passed to Final Response Agent

The following synthesized information from all layers, along with the original user prompt, was passed to the final response agent ({final_agent_name} - `{final_agent_model}`). The final agent used this information to generate the final MoA response.
"""
    for layer in layer_details:
        layer_num = layer["layer_number"]
        markdown_content += f"""
### Layer {layer_num} Synthesis

{sanitize_for_markdown(layer.get("synthesis", "N/A"))}

### Layer {layer_num} Devil's Advocate

{sanitize_for_markdown(layer.get("devils_advocate", "N/A"))}

---
"""

    markdown_content += f"""
## Final MoA Response
**Final Response Agent:** {final_agent_name} - `{final_agent_model}`

{sanitize_for_markdown(final_response)}
"""

    return markdown_content.strip()