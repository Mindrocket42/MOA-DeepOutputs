#!/usr/bin/env python3
"""
Basic test script for Single Agent Workflow implementation.

This script tests the core functionality without making actual API calls.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")

    try:
        # Test core imports
        from single_agent.single_agent_engine import (
            SingleAgentWorkflow, SingleAgentTracer,
            SingleAgent, OpenRouterSingleAgent, SingleAgentGenerationError,
            SA_RESPONSE_AGENT_MODEL, SA_NUM_LAYERS, logger
        )
        print("PASS: Core imports successful")

        # Test agent imports
        from single_agent.single_agent_engine.agents import SingleAgent, OpenRouterSingleAgent
        print("PASS: Agent imports successful")

        # Test utility imports
        from single_agent.single_agent_engine.utils import sanitize_filename, format_timestamp_for_filename
        print("PASS: Utility imports successful")

        # Test prompt imports
        from single_agent.single_agent_engine.prompts import build_response_prompt, build_devils_advocate_prompt
        print("PASS: Prompt imports successful")

        return True

    except ImportError as e:
        print(f"FAIL: Import failed: {e}")
        return False

def test_configuration():
    """Test that configuration is loaded correctly."""
    print("\nTesting configuration...")

    try:
        from single_agent.single_agent_engine.config import (
            SA_RESPONSE_AGENT_MODEL, SA_DEVILS_ADVOCATE_AGENT_MODEL,
            SA_SYNTHESIS_AGENT_MODEL, SA_FINAL_AGENT_MODEL,
            SA_NUM_LAYERS, SA_ENABLE_SELF_REVIEW
        )

        # Check that models are configured
        assert SA_RESPONSE_AGENT_MODEL, "Response agent model not configured"
        assert SA_DEVILS_ADVOCATE_AGENT_MODEL, "Devils advocate model not configured"
        assert SA_SYNTHESIS_AGENT_MODEL, "Synthesis agent model not configured"
        assert SA_FINAL_AGENT_MODEL, "Final agent model not configured"

        # Check that layers is reasonable
        assert SA_NUM_LAYERS > 0, "Number of layers must be positive"

        print(f"PASS: Configuration loaded: {SA_NUM_LAYERS} layers, self-review: {SA_ENABLE_SELF_REVIEW}")
        print(f"   Response: {SA_RESPONSE_AGENT_MODEL}")
        print(f"   Devils Advocate: {SA_DEVILS_ADVOCATE_AGENT_MODEL}")
        print(f"   Synthesis: {SA_SYNTHESIS_AGENT_MODEL}")
        print(f"   Final: {SA_FINAL_AGENT_MODEL}")

        return True

    except Exception as e:
        print(f"FAIL: Configuration test failed: {e}")
        return False

def test_prompt_building():
    """Test that prompts can be built successfully."""
    print("\nTesting prompt building...")

    try:
        from single_agent.single_agent_engine.prompts import (
            build_response_prompt, build_self_review_prompt,
            build_devils_advocate_prompt, build_synthesis_prompt, build_final_prompt
        )

        test_prompt = "What is the best way to learn Python?"
        test_response = "Learning Python through practice and projects is most effective."

        # Test response prompt
        response_prompt = build_response_prompt(test_prompt)
        assert "expert Response Agent" in response_prompt
        assert test_prompt in response_prompt
        print("PASS: Response prompt building works")

        # Test self-review prompt
        review_prompt = build_self_review_prompt(test_prompt, test_response)
        assert "self-review" in review_prompt.lower()
        assert test_response in review_prompt
        print("PASS: Self-review prompt building works")

        # Test devils advocate prompt
        devils_prompt = build_devils_advocate_prompt(test_prompt, test_response)
        assert "devils advocate" in devils_prompt.lower()
        assert "stress-test" in devils_prompt.lower()
        print("PASS: Devils advocate prompt building works")

        # Test synthesis prompt
        synthesis_prompt = build_synthesis_prompt(test_prompt, test_response, "Some critique here")
        assert "synthesis agent" in synthesis_prompt.lower()
        assert "### key dissenting points" in synthesis_prompt.lower()
        print("PASS: Synthesis prompt building works")

        # Test final prompt
        layer_outputs = [{"response": test_response, "devils_advocate": "Some critique"}]
        final_prompt = build_final_prompt(test_prompt, layer_outputs)
        assert "final agent" in final_prompt.lower()
        assert "authoritative decision" in final_prompt.lower()
        print("PASS: Final prompt building works")

        return True

    except Exception as e:
        print(f"FAIL: Prompt building test failed: {e}")
        return False

def test_agent_instantiation():
    """Test that agents can be instantiated (without API calls)."""
    print("\nTesting agent instantiation...")

    try:
        from single_agent.single_agent_engine.agents import OpenRouterSingleAgent

        # Test response agent
        response_agent = OpenRouterSingleAgent(
            name="Test Response Agent",
            model="test-model",
            role="Test Role",
            cognitive_function="response"
        )
        assert response_agent.name == "Test Response Agent"
        assert response_agent.cognitive_function == "response"
        print("PASS: Response agent instantiation works")

        # Test devils advocate agent
        devils_agent = OpenRouterSingleAgent(
            name="Test Devils Advocate",
            model="test-model",
            role="Test Role",
            cognitive_function="devils_advocate"
        )
        assert devils_agent.cognitive_function == "devils_advocate"
        print("PASS: Devils advocate agent instantiation works")

        return True

    except Exception as e:
        print(f"FAIL: Agent instantiation test failed: {e}")
        return False

def test_workflow_instantiation():
    """Test that workflow can be instantiated."""
    print("\nTesting workflow instantiation...")

    try:
        from single_agent.single_agent_engine.workflow import SingleAgentWorkflow

        # Test workflow instantiation (without tracer for basic test)
        workflow = SingleAgentWorkflow(tracer=None)

        # Check that agents are created
        assert hasattr(workflow, 'response_agent')
        assert hasattr(workflow, 'devils_advocate_agent')
        assert hasattr(workflow, 'synthesis_agent')
        assert hasattr(workflow, 'final_agent')

        # Check agent cognitive functions
        assert workflow.response_agent.cognitive_function == "response"
        assert workflow.devils_advocate_agent.cognitive_function == "devils_advocate"
        assert workflow.synthesis_agent.cognitive_function == "synthesis"
        assert workflow.final_agent.cognitive_function == "final"

        print("PASS: Workflow instantiation works")
        print(f"   Response Agent: {workflow.response_agent.model}")
        print(f"   Devils Advocate: {workflow.devils_advocate_agent.model}")
        print(f"   Synthesis Agent: {workflow.synthesis_agent.model}")
        print(f"   Final Agent: {workflow.final_agent.model}")

        return True

    except Exception as e:
        print(f"FAIL: Workflow instantiation test failed: {e}")
        return False

def test_directory_structure():
    """Test that directory structure exists."""
    print("\nTesting directory structure...")

    required_dirs = [
        "single_agent/single_agent_engine",
        "single_agent/single_agent_engine/agents",
        "single_agent/reports",
        "single_agent/traces",
        "single_agent/prompts",
        "single_agent/dry_runs/reports",
        "single_agent/dry_runs/traces"
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"FAIL: Missing directories: {missing_dirs}")
        return False

    print("PASS: All required directories exist")
    return True

def main():
    """Run all tests."""
    print("TEST: Single Agent Workflow - Basic Functionality Tests")
    print("=" * 60)

    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Prompt Building", test_prompt_building),
        ("Agent Instantiation", test_agent_instantiation),
        ("Workflow Instantiation", test_workflow_instantiation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"FAIL: {test_name} test failed")
        except Exception as e:
            print(f"FAIL: {test_name} test failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: All tests passed! Single Agent implementation is ready.")
        return 0
    else:
        print("WARNING: Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())