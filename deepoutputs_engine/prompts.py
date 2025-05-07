"""
Prompt construction logic for each agent phase.
"""

from typing import Any, Dict, List

def build_layer_prompt(original_prompt: str, prev_synthesis: str, prev_devils_advocate: str, layer_index: int) -> str:
    """Creates a streamlined, LLM-friendly prompt for agents in a specific layer."""
    if layer_index == 0:
        # Add clear instructions for the first layer
        return f"""You are an expert AI agent. Your task is to answer the following user prompt as clearly and insightfully as possible, using sound reasoning and, if relevant, calculations or examples.

User Prompt:
{original_prompt}

Please provide a well-structured, direct answer. If there are ambiguities, state your assumptions."""
    else:
        context_header = f"--- Previous Layer Context (Layer {layer_index}) ---"
        synthesis_text = f"Synthesis (summary so far):\n{prev_synthesis}\n" if prev_synthesis else ""
        critique_text = f"Devil's Advocate Critique:\n{prev_devils_advocate}\n" if prev_devils_advocate else ""
        context_block = f"{context_header}\n\n{synthesis_text}{critique_text}---\n\n" if (synthesis_text or critique_text) else ""
        return f"""You are an expert AI agent. Your task is to answer the user's original prompt, taking into account the analysis and critique from the previous layer.

User Prompt:
{original_prompt}

{context_block}Instructions for Layer {layer_index + 1}:
- Carefully read the user prompt.
- Consider the synthesis and critique above (if present) as context.
- Generate your own independent answer to the user prompt, not just a rephrasing of prior outputs.
- Briefly explain if/how the previous context changed your answer, or if you disagree with it.
- Ensure your answer is clear, well-reasoned, and addresses the main question.
"""

def build_aggregation_prompt(original_prompt: str, current_layer_prompt: str, initial_responses: List[str]) -> str:
    """Creates the prompt for the aggregation/peer review step."""
    response_text = "\n\n---\n\n".join([f"Response from Agent {i+1}:\n{resp}" for i, resp in enumerate(initial_responses)])

    return f"""Original User Prompt:
{original_prompt}

---
Current Layer Context/Prompt Given to Agents:
{current_layer_prompt}
---

Initial Responses Received in this Layer:
{response_text}
---

Your Task (Aggregation & Peer Review):
You are reviewing the initial responses above to the challenge posed by the 'Current Layer Context/Prompt'.

1.  **Critique All Responses:** Analyze the logic, reasoning, and factual accuracy of *all* responses provided, including potentially your own if you recognize it. Be specific in your critiques.
2.  **Identify Assumptions:** Explicitly list and challenge the key assumptions (stated or unstated) made in *each* response.
3.  **Verify (If Applicable):** If the prompt involves calculations or checkable facts, attempt to verify them. State your findings.
4.  **Explore Alternatives:** Consider alternative interpretations of the prompt or fundamentally different approaches that were missed. Could the reasoning be flawed?
5.  **Synthesize Strengths/Weaknesses:** Briefly summarize the strongest points and biggest weaknesses you observed across all responses.
6.  **Generate Improved Response:** Based *only* on your critical analysis and understanding of the Original User Prompt, provide your own improved and independent answer to the core question asked in the Original User Prompt. Do NOT simply combine the previous answers. Use them as context for critique but form your own conclusion.
7.  **Explain Your Reasoning:** Justify why your improved response is better or more accurate than the initial responses. If you agree with parts of other responses, state which and why.

Structure your output clearly, addressing each point above. Mark your final improved response clearly (e.g., "My Improved Response:").
"""

def build_synthesis_prompt(original_prompt: str, current_layer_prompt: str, aggregated_responses: List[str]) -> str:
    """Creates the prompt for the synthesis agent."""
    response_text = "\n\n---\n\n".join([f"Aggregated Response from Agent {i+1}:\n{resp}" for i, resp in enumerate(aggregated_responses)])

    return f"""Original User Prompt:
{original_prompt}

---
Current Layer Context/Prompt Given to Agents:
{current_layer_prompt}
---

Aggregated/Reviewed Responses Received in this Layer:
{response_text}
---

Your Task (Synthesis Agent):
---
Analyze the full text of the "Aggregated/Reviewed Responses" provided above and deliver one unified, richly detailed synthesis that sets the stage for the next phase of analysis. Your output should include:

1. **Core Insights:**  
   - Thoroughly extract and explain the primary conclusions, findings, and proposed solutions.  
   - Illustrate how these insights interconnect or build on each other.

2. **Consensus & Divergence:**  
   - Map out where the responses strongly agree—citing specific points or language—and where they diverge or present conflicting perspectives.  
   - For each major disagreement, briefly describe the reasoning on each side.

3. **Confidence Levels & Uncertainties:**  
   - Highlight areas where the aggregated responses express high confidence (e.g., recurring evidence, consistent recommendations).  
   - Pinpoint concepts or data points that remain ambiguous, under‑supported, or contested.

4. **Outstanding Questions & Gaps:**  
   - Identify any aspects of the Original User Prompt that still lack clarity or need deeper probing.  
   - Formulate explicit questions or objectives that the next analytical layer should address.

5. **Expansive Synthesis Narrative:**  
   - In a multi‑paragraph narrative (not limited to two or three), weave together the above elements into a coherent story of what's known, what's debated, and what's missing.  
   - Use transitions and headings (if helpful) to guide the reader through each section's logical flow.

6. **Next‑Layer Roadmap:**  
   - Conclude with a detailed outline of the next analytical steps—methodologies, data sources, or stakeholder inputs—that will resolve the open questions and drive toward a final answer.

**Do NOT** simply list bullet points. Craft a prose narrative that captures both the nuance and the breadth of the discourse, fully preparing readers for the deeper dive to come.
"""

def build_devils_advocate_prompt(original_prompt: str, current_layer_prompt: str, aggregated_responses: List[str]) -> str:
    """Creates the prompt for the devil's advocate agent."""
    response_text = "\n\n---\n\n".join([f"Aggregated Response from Agent {i+1}:\n{resp}" for i, resp in enumerate(aggregated_responses)])

    return f"""Original User Prompt:
{original_prompt}

---
Current Layer Context/Prompt Given to Agents:
{current_layer_prompt}
---

Aggregated/Reviewed Responses Received in this Layer:
{response_text}
---

Your Task (Aggressive Devil's Advocate):
Your sole purpose is to rigorously challenge the prevailing conclusions and reasoning found in the 'Aggregated/Reviewed Responses'. Be critical, skeptical, and look for flaws.

1.  **Attack the Consensus:** If there's a common answer or approach, argue forcefully against it. Why might it be wrong, incomplete, or based on flawed premises?
2.  **Challenge Fundamental Assumptions:** Question the most basic assumptions made by the agents. Are they justified? What if they are wrong?
3.  **Identify Blind Spots:** What critical factors, alternative scenarios, edge cases, or potential negative consequences have *all* the responses overlooked?
4.  **Expose Logical Fallacies:** Point out any weak arguments, leaps in logic, or inconsistencies within or between the responses.
5.  **Propose Contrarian Views:** Offer at least one completely different perspective or solution, even if it seems unconventional. Explain why it *could* be valid.
6.  **Nitpick Calculations/Data (If Applicable):** If there are numbers or data, question their validity, source, or interpretation.

Your tone should be challenging and critical. Your goal is NOT to be helpful or find agreement, but to stress-test the current conclusions by finding their weakest points. Structure your critique logically. Start your response directly with the critique, without introductory phrases like "Here is the Devil's Advocate perspective:".
"""

def build_final_prompt(original_prompt: str, layer_details: List[Dict[str, Any]]) -> str:
    """Generates the final response based on all layer outputs."""
    consolidated_context = ""
    for layer in layer_details:
        consolidated_context += f"--- Layer {layer['layer_number']} ---\n"
        consolidated_context += f"Synthesis:\n{layer['synthesis']}\n\n"
        consolidated_context += f"Devil's Advocate Critique:\n{layer['devils_advocate']}\n\n"

    return f"""Original User Prompt:
{original_prompt}

---
Consolidated Context from All Layers:
{consolidated_context}
---

Your Task (Final Agent):
You must provide the definitive, final answer to the 'Original User Prompt'. Use the 'Consolidated Context' as input and analysis history, but rely ultimately on your own reasoning and interpretation of the original prompt.

1.  **Re-evaluate Prompt:** Carefully re-read and interpret the 'Original User Prompt'. What is the core question or task?
2.  **Cross-Check Context:** Review the 'Consolidated Context'. Identify the key insights, conclusions, and critiques from the layers. Note areas of agreement and disagreement.
3.  **Identify Discrepancies:** Explicitly state any major discrepancies, contradictions, or unresolved issues between the layers or between the context and the 'Original User Prompt'.
4.  **Resolve Conflicts:** Address the discrepancies you identified. Explain how you are resolving them based on your understanding of the original prompt and logical reasoning. State which arguments or pieces of information you are prioritizing and why.
5.  **Formulate Final Answer:** Provide a clear, comprehensive, and well-reasoned final answer directly addressing the 'Original User Prompt'. Structure your answer logically.
6.  **Justify Your Answer:** Briefly explain why your final answer is the most accurate and complete response, referencing how you used or discarded elements from the context. If you disagree strongly with the context, state why.
7.  **Acknowledge Uncertainty (If Any):** If ambiguity remains or multiple valid answers exist, state this clearly and explain the different possibilities or necessary assumptions.

Ensure your final output is coherent, directly answers the user's original question, and represents your best possible analysis. Start your response directly with the final answer, without introductory phrases like "Here is the final answer:".
"""