"""
Prompt construction logic for each agent phase – v2 (2025‑07‑05).

Key improvements
----------------
* Language tightened for clarity and consistency.
* Devil's‑Advocate signal elevated – every downstream stage must surface
  dissent either as critique **and** as a self‑contained "Dissenting Viewpoint".
* Prompts now explicitly ask agents to tag assumptions and confidence levels.
* No function signatures changed – workflow compatibility preserved.
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Helper builders – layer prompts
# ---------------------------------------------------------------------------

def build_layer_prompt(
    original_prompt: str,
    prev_synthesis: str,
    prev_devils_advocate: str,
    layer_index: int,
) -> str:
    """Create the per‑agent prompt for a given analytical layer.

    The first layer receives the raw user prompt. Subsequent layers are passed a
    compact context header containing *both* the running synthesis and the
    Devil's‑Advocate critique so far. Agents are required to reference – but not
    merely parrot – this context, and to state explicitly whether they agree or
    dissent.
    """
    if layer_index == 0:
        return (
            f"""You are an expert AI agent. Produce the clearest, most insightful
answer you can to the user prompt below. Use tight logic, cite concrete
examples or calculations where helpful. If ambiguity exists, declare your
assumptions.

User Prompt:
{original_prompt}
"""
        )

    # Build context for layers > 0
    header = f"--- Previous‑Layer Context (Layer {layer_index}) ---"
    synthesis_block = (
        f"Running Synthesis So Far:\n{prev_synthesis}\n" if prev_synthesis else ""
    )
    critique_block = (
        f"Devil's‑Advocate Critique:\n{prev_devils_advocate}\n" if prev_devils_advocate else ""
    )
    context = f"{header}\n\n{synthesis_block}{critique_block}---\n\n" if (
        synthesis_block or critique_block
    ) else ""

    return (
        f"""You are an expert AI agent. Address the *original* user prompt while
engaging critically with the context provided.

User Prompt:
{original_prompt}

{context}Instructions for Layer {layer_index + 1}:
● Reflect on how the synthesis and critique influence your stance. State where
  you **agree**, **expand**, or **disagree** – be explicit.
● Provide an independent answer – do **not** merely rephrase prior text.
● Tag any key *assumptions* you make and assign a 1‑line *confidence rating*
  (High / Medium / Low).
"""
    )


# ---------------------------------------------------------------------------
# Aggregation / peer‑review prompt
# ---------------------------------------------------------------------------

def build_aggregation_prompt(
    original_prompt: str,
    current_layer_prompt: str,
    initial_responses: List[str],
) -> str:
    """Prompt for the aggregation‑and‑review agent."""

    response_blob = "\n\n---\n\n".join(
        [f"Response from Agent {i + 1}:\n{resp}" for i, resp in enumerate(initial_responses)]
    )

    return (
        f"""Original User Prompt:
{original_prompt}

---
Prompt Shown to This Layer's Agents:
{current_layer_prompt}
---

Initial Responses Produced:
{response_blob}
---

Your Task – Aggregation & Peer Review
====================================
1. **Critique Each Response** – logic, evidence, clarity.
2. **Surface Assumptions** – catalogue explicit *and* hidden premises.
3. **Check Facts / Maths** – verify any numbers or citations referenced.
4. **Spot Blind‑Spots** – name missing angles or alternative frames.
5. **Compare & Contrast** – who agrees, who diverges; where & why.
6. **Draft Your *Improved* Answer** – fresh, independent; build on strengths,
   correct weaknesses; keep it concise yet thorough.
7. **Explain Superiority** – why yours beats the pack.

*Output structure:*
• Use clear section headings. End with **My Improved Response:** followed by the
  refined answer.
"""
    )


# ---------------------------------------------------------------------------
# Synthesis prompt – now with dissent slot
# ---------------------------------------------------------------------------

def build_synthesis_prompt(
    original_prompt: str,
    current_layer_prompt: str,
    aggregated_responses: List[str],
) -> str:
    """Prompt for the synthesis agent combining peer‑reviewed outputs."""

    agg_blob = "\n\n---\n\n".join(
        [f"Aggregated Response {i + 1}:\n{resp}" for i, resp in enumerate(aggregated_responses)]
    )

    return (
        f"""Original User Prompt:
{original_prompt}

---
Prompt to Agents in This Layer:
{current_layer_prompt}
---

Aggregated & Reviewed Responses:
{agg_blob}
---

Your Task – Synthesis Agent
===========================
Generate a single, coherent synthesis that does **three** things:
1. **Unified Narrative** – integrate the best insights, clearly labelled.
2. **Dissenting Viewpoint** – summarise the strongest Devil's‑Advocate or other
   minority critiques *in a dedicated boxed section* so that they are impossible
   to ignore later.
3. **Roadmap Forward** – outline unanswered questions, data needs, or next
   investigative angles.

Structure: prose narrative with sub‑headings. Make dissent section visually
obvious, e.g. prefix with "### Dissenting Viewpoint".
"""
    )


# ---------------------------------------------------------------------------
# Devil's‑Advocate prompt – unchanged tone, reinforced output tag
# ---------------------------------------------------------------------------

def build_devils_advocate_prompt(
    original_prompt: str,
    current_layer_prompt: str,
    aggregated_responses: List[str],
) -> str:
    """Prompt the aggressive Devil's‑Advocate agent."""

    agg_blob = "\n\n---\n\n".join(
        [f"Aggregated Response {i + 1}:\n{resp}" for i, resp in enumerate(aggregated_responses)]
    )

    return (
        f"""Original User Prompt:
{original_prompt}

---
Prompt Shown to Agents:
{current_layer_prompt}
---

Aggregated & Reviewed Responses:
{agg_blob}
---

Your Task – **Aggressive Devil's‑Advocate**
==========================================
Your job is to *stress‑test* the consensus. Attack hard:
• **Undermine Consensus** – why might it fail?
• **Challenge Core Assumptions** – expose shaky ground.
• **Highlight Blind‑Spots & Edge‑Cases** – what's missing?
• **Call Out Logical Fallacies** – name them.
• **Offer One Plausible Contrarian Solution** – even if radical.

Begin immediately with critique bullets – no introduction fluff. Conclude with a
short "**If I Had to Bet:**" paragraph stating which single point you think
would most likely *break* if implemented.
"""
    )


# ---------------------------------------------------------------------------
# Final agent prompt – must surface dissent visibly
# ---------------------------------------------------------------------------

def build_final_prompt(
    original_prompt: str,
    layer_details: List[Dict[str, Any]],
) -> str:
    """Construct the prompt for the final answering agent."""

    consolidated = ""
    for layer in layer_details:
        consolidated += (
            f"--- Layer {layer['layer_number']} ---\n"
            f"Synthesis:\n{layer['synthesis']}\n\n"
            f"Devil's‑Advocate:\n{layer['devils_advocate']}\n\n"
        )

    return (
        f"""Original User Prompt:
{original_prompt}

---
All‑Layer Context:
{consolidated}
---

Your Task – **Final Agent**
==========================
Deliver the authoritative answer to the Original User Prompt.

Steps to Follow *within your own head* (do NOT output these as steps):
1. Parse the prompt – what exactly is being asked?
2. Absorb the layered context – note agreements, contradictions, dissent.
3. Decide what evidence/reasoning to keep or discard.
4. Form your own top‑down answer.

**Output Requirements**
-----------------------
A. **Direct Answer** – start immediately.
B. **Why This Answer** – brief justification referencing context layers.
C. **Dissenting Viewpoint (if materially different)** – one concise paragraph
   capturing the strongest Devil's‑Advocate critique so decision‑makers can see
   it surfaced.
D. **Residual Uncertainty** – if any.

No headings like "Final Answer:" – just start.
"""
    )
