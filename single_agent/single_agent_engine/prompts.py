"""
Prompt construction logic for single agent workflow phases.

This module provides specialized prompts for each cognitive function in the
single agent workflow: Response, Self-Review, Devils Advocate, Synthesis, and Final.

Key features:
- Cognitive function specialization
- Context accumulation across phases
- Dissent preservation and surfacing
- Layer-aware prompt building
- Assumption tagging and confidence levels
"""

from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# Response Agent Prompts
# ---------------------------------------------------------------------------

def build_response_prompt(
    original_prompt: str,
    layer_index: int = 0,
    prev_synthesis: str = "",
    prev_devils_advocate: str = ""
) -> str:
    """
    Build prompt for the Response Agent - initial analysis and reasoning.

    Args:
        original_prompt: The user's original prompt
        layer_index: Current layer number (0-based)
        prev_synthesis: Previous layer's synthesis (if any)
        prev_devils_advocate: Previous layer's devils advocate critique (if any)

    Returns:
        Formatted prompt for response agent
    """
    if layer_index == 0:
        return f"""You are an expert Response Agent. Your role is to provide deep, thoughtful analysis and reasoning.

**Task**: Produce a comprehensive response to the user prompt below. Focus on:
- Clear, logical reasoning with concrete examples
- Identification of key assumptions and their validity
- Balanced consideration of multiple perspectives
- Practical implications and actionable insights

**Guidelines**:
- Be thorough but concise - quality over quantity
- Tag any assumptions you make with [ASSUMPTION: ...]
- Include confidence levels: [HIGH/MEDIUM/LOW CONFIDENCE]
- Consider edge cases and limitations
- Provide evidence or reasoning for your conclusions

**User Prompt**:
{original_prompt}

**Response**:"""

    # Layer > 0: Include context from previous layers
    context_parts = []
    if prev_synthesis:
        context_parts.append(f"**Previous Layer Synthesis**:\n{prev_synthesis}")
    if prev_devils_advocate:
        context_parts.append(f"**Previous Layer Critique**:\n{prev_devils_advocate}")

    context = "\n\n".join(context_parts) if context_parts else ""

    return f"""You are an expert Response Agent analyzing a complex problem across multiple layers.

**Original User Prompt**:
{original_prompt}

{context}

**Task (Layer {layer_index + 1})**:
Build upon the previous layer's analysis while addressing the identified critiques. Focus on:
- How the previous critique affects your current analysis
- Whether you agree, disagree, or expand on prior conclusions
- New insights gained from the layered analysis
- Updated assumptions and confidence levels

**Guidelines**:
- Reference specific points from the previous critique
- State explicitly where you [AGREE], [DISAGREE], or [EXPAND]
- Update confidence levels based on new information
- Maintain balance between previous insights and new perspectives

**Response**:"""

def build_self_review_prompt(
    original_prompt: str,
    initial_response: str,
    layer_index: int = 0
) -> str:
    """
    Build prompt for self-review phase - critical assessment of initial response.

    Args:
        original_prompt: The user's original prompt
        initial_response: The response agent's initial response
        layer_index: Current layer number (0-based)

    Returns:
        Formatted prompt for self-review
    """
    return f"""You are conducting a rigorous self-review of your initial response.

**Original User Prompt**:
{original_prompt}

**Your Initial Response**:
{initial_response}

**Self-Review Task**:
Critically assess your initial response by asking:
1. **Logic & Reasoning**: Are the arguments sound? Any logical fallacies?
2. **Evidence & Support**: Are claims sufficiently supported? Missing evidence?
3. **Assumptions**: Are assumptions clearly stated and reasonable?
4. **Completeness**: Any important angles or perspectives missing?
5. **Clarity**: Is the response clear and well-structured?
6. **Balance**: Does it consider alternative viewpoints adequately?

**Output Format**:
- **Strengths**: What works well in the response
- **Weaknesses**: Specific areas needing improvement
- **Gaps**: Missing information or perspectives
- **Recommendations**: How to improve the response
- **Confidence Assessment**: Overall confidence level and why

**Self-Review**:"""

# ---------------------------------------------------------------------------
# Devils Advocate Agent Prompts
# ---------------------------------------------------------------------------

def build_devils_advocate_prompt(
    original_prompt: str,
    response: str,
    self_review: str = "",
    layer_index: int = 0,
    prev_synthesis: str = ""
) -> str:
    """
    Build prompt for Devils Advocate Agent - aggressive critical challenge.

    Args:
        original_prompt: The user's original prompt
        response: The response agent's output
        self_review: Self-review assessment (if available)
        layer_index: Current layer number (0-based)
        prev_synthesis: Previous layer's synthesis (if any)

    Returns:
        Formatted prompt for devils advocate
    """
    context_parts = [
        f"**Original User Prompt**:\n{original_prompt}",
        f"**Response Agent Output**:\n{response}"
    ]

    if self_review:
        context_parts.append(f"**Self-Review Assessment**:\n{self_review}")

    if prev_synthesis:
        context_parts.append(f"**Previous Layer Context**:\n{prev_synthesis}")

    context = "\n\n".join(context_parts)

    return f"""{context}

**Your Role**: You are the Devils Advocate Agent - an aggressive critical challenger.

**Task**: Stress-test this response with maximum skepticism. Your goal is to identify flaws, assumptions, and potential failure points.

**Challenge Guidelines**:
1. **Undermine Core Assumptions**: Question fundamental premises
2. **Highlight Blind Spots**: What important perspectives are missing?
3. **Expose Weak Logic**: Find logical fallacies or weak reasoning
4. **Identify Edge Cases**: When might this approach fail?
5. **Challenge Evidence**: Is the supporting evidence sufficient?
6. **Alternative Solutions**: Propose radically different approaches

**Output Requirements**:
- Start immediately with critique bullets - no introduction
- Be specific and evidence-based in your challenges
- End with: "**Most Critical Issue**: [single most important problem]"

**Devils Advocate Critique**:"""

# ---------------------------------------------------------------------------
# Synthesis Agent Prompts
# ---------------------------------------------------------------------------

def build_synthesis_prompt(
    original_prompt: str,
    response: str,
    devils_advocate: str,
    self_review: str = "",
    layer_index: int = 0,
    prev_synthesis: str = ""
) -> str:
    """
    Build prompt for Synthesis Agent - integration with dissent management.

    Args:
        original_prompt: The user's original prompt
        response: Response agent's output
        devils_advocate: Devils advocate critique
        self_review: Self-review assessment (if available)
        layer_index: Current layer number (0-based)
        prev_synthesis: Previous layer's synthesis (if any)

    Returns:
        Formatted prompt for synthesis agent
    """
    context_parts = [
        f"**Original User Prompt**:\n{original_prompt}",
        f"**Response Agent Analysis**:\n{response}",
        f"**Devils Advocate Critique**:\n{devils_advocate}"
    ]

    if self_review:
        context_parts.append(f"**Self-Review Assessment**:\n{self_review}")

    if prev_synthesis:
        context_parts.append(f"**Previous Layer Synthesis**:\n{prev_synthesis}")

    context = "\n\n".join(context_parts)

    return f"""{context}

**Your Role**: You are the Synthesis Agent - integrator of insights and dissent manager.

**Task**: Create a unified synthesis that captures the best of all perspectives while preserving important dissent.

**Synthesis Requirements**:
1. **Unified Narrative**: Integrate strongest insights from all sources
2. **Dissent Preservation**: Surface the most important Devils Advocate critiques in a clearly marked section
3. **Resolution Assessment**: Where possible, address or acknowledge critiques
4. **Confidence Calibration**: Adjust confidence levels based on challenges raised
5. **Forward Roadmap**: Identify remaining questions or data needs

**Output Structure**:
- **Core Synthesis**: Main integrated analysis
- **### Key Dissenting Points**: Clearly marked section with strongest critiques
- **Updated Assessment**: How the synthesis addresses or acknowledges dissent
- **Remaining Uncertainty**: Areas where confidence is reduced

**Synthesis**:"""

# ---------------------------------------------------------------------------
# Final Agent Prompts
# ---------------------------------------------------------------------------

def build_final_prompt(
    original_prompt: str,
    layer_outputs: List[Dict[str, Any]]
) -> str:
    """
    Build prompt for Final Agent - authoritative decision with dissent surfacing.

    Args:
        original_prompt: The user's original prompt
        layer_outputs: List of outputs from all layers

    Returns:
        Formatted prompt for final agent
    """
    # Build consolidated context from all layers
    consolidated_parts = []

    for i, layer in enumerate(layer_outputs):
        layer_num = i + 1
        consolidated_parts.append(f"--- LAYER {layer_num} ---")

        if "response" in layer:
            consolidated_parts.append(f"**Response Agent**: {layer['response']}")
        if "self_review" in layer:
            consolidated_parts.append(f"**Self-Review**: {layer['self_review']}")
        if "devils_advocate" in layer:
            consolidated_parts.append(f"**Devils Advocate**: {layer['devils_advocate']}")
        if "synthesis" in layer:
            consolidated_parts.append(f"**Synthesis**: {layer['synthesis']}")

    consolidated_context = "\n\n".join(consolidated_parts)

    return f"""**Original User Prompt**:
{original_prompt}

**Consolidated Analysis from All Layers**:
{consolidated_context}

**Your Role**: You are the Final Agent - authoritative decision maker.

**Task**: Deliver the final, authoritative answer to the original user prompt.

**Final Decision Process** (think through this internally):
1. Parse the original prompt - what exactly is being asked?
2. Absorb all layer context - note agreements, contradictions, dissent
3. Decide what evidence/reasoning to keep or discard
4. Form your own top-down final answer

**Output Requirements**:
- **Direct Answer**: Start immediately with your final conclusion
- **Why This Answer**: Brief justification referencing layer context
- **Dissenting Viewpoint**: One concise paragraph capturing strongest Devils Advocate critiques
- **Residual Uncertainty**: Any remaining doubts or limitations

**Final Answer**:"""