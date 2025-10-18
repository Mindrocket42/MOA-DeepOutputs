"""
Tracing system for Single Agent Workflow.

Adapted from MOA tracing system to track single agent cognitive functions,
layered processing, and dissent management.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict

from .config import SA_TRACE_DIR, logger

class SingleAgentTracer:
    """
    Enhanced tracer for single agent workflow logging and metrics.

    Tracks cognitive function transitions, layer processing, dissent management,
    and performance metrics specific to single agent workflows.
    """

    def __init__(self, run_id: str, enabled: bool = True):
        """
        Initialize the single agent tracer.

        Args:
            run_id: Unique identifier for this workflow run
            enabled: Whether tracing is enabled
        """
        self.enabled = enabled
        self.run_id = run_id
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.base_dir = Path(SA_TRACE_DIR)
        self.json_dir = self.base_dir / "json" / run_id
        self.json_path = self.json_dir / f"trace_{self.timestamp}.jsonl"
        self.json_file = None

        # Initialize metrics tracking for single agent workflow
        self.metrics = {
            "api_calls": defaultdict(int),
            "cognitive_function_calls": defaultdict(int),
            "token_usage": defaultdict(int),
            "latency": defaultdict(list),
            "errors": defaultdict(int),
            "dissent_points": [],
            "confidence_levels": []
        }

        # Initialize performance tracking
        self.performance = {
            "start_time": time.time(),
            "layer_times": [],
            "phase_times": defaultdict(list),
            "total_tokens": 0,
            "cognitive_transitions": []
        }

        if enabled:
            try:
                self.json_dir.mkdir(parents=True, exist_ok=True)
                self.json_file = open(self.json_path, "a", encoding="utf-8")
                logger.info(f"Single Agent Tracer initialized with run_id: {run_id}")
                logger.info(f"Trace file: {self.json_path}")
            except Exception as e:
                logger.error(f"Failed to initialize Single Agent Tracer: {str(e)}")
                self.enabled = False
                raise

    def log(self, event: Dict[str, Any], level: str = "info") -> None:
        """
        Log an event to JSONL file.

        Args:
            event: Dictionary containing event data
            level: Log level (debug, info, warning, error)
        """
        if not self.enabled:
            return

        try:
            # Add metadata
            event['timestamp'] = datetime.now().isoformat()
            event['level'] = level
            event['run_id'] = self.run_id

            # Write JSONL
            self.json_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.json_file.flush()

            # Log to console
            log_msg = f"Event logged: {event.get('event', 'Unknown Event')}"
            if level == "error":
                logger.error(log_msg)
            elif level == "warning":
                logger.warning(log_msg)
            elif level == "debug":
                logger.debug(log_msg)
            else:
                logger.info(log_msg)

        except Exception as e:
            logger.error(f"Failed to log event: {str(e)}")
            if not self.enabled:
                return
            raise

    def log_api_call(self,
                     model: str,
                     prompt: str,
                     response: str,
                     duration: float,
                     tokens_used: Optional[int] = None,
                     cognitive_function: Optional[str] = None,
                     error: Optional[str] = None) -> None:
        """
        Log an API call with single agent specific metadata.

        Args:
            model: Name of the model used
            prompt: Input prompt
            response: Model response
            duration: Call duration in seconds
            tokens_used: Number of tokens used
            cognitive_function: Cognitive function being performed
            error: Error message if any
        """
        # Update metrics
        self.metrics["api_calls"][model] += 1
        if cognitive_function:
            self.metrics["cognitive_function_calls"][cognitive_function] += 1
        if tokens_used:
            self.metrics["token_usage"][model] += tokens_used
            self.performance["total_tokens"] += tokens_used
        self.metrics["latency"][model].append(duration)
        if error:
            self.metrics["errors"][model] += 1

        event = {
            "event": "API Call",
            "model": model,
            "cognitive_function": cognitive_function,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "duration": duration,
            "tokens_used": tokens_used,
            "error": error
        }

        self.log(event)

    def log_workflow_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log workflow-level events.

        Args:
            event_type: Type of workflow event
            data: Event data
        """
        event = {
            "event": f"Workflow {event_type}",
            **data
        }
        self.log(event)

    def log_layer_event(self,
                       layer_num: int,
                       event_type: str,
                       prompt: str,
                       metrics: Dict[str, Any]) -> None:
        """
        Log layer processing events.

        Args:
            layer_num: Layer number
            event_type: Start/End/Error
            prompt: Current prompt being processed
            metrics: Performance metrics
        """
        event = {
            "event": f"Layer {event_type}",
            "layer_number": layer_num,
            "prompt_length": len(prompt),
            **metrics
        }

        if event_type == "End":
            self.performance["layer_times"].append(metrics.get("duration", 0))

        self.log(event)

    def log_phase_event(self,
                       phase: str,
                       layer_num: int,
                       event_type: str,
                       duration: Optional[float] = None,
                       cognitive_function: Optional[str] = None) -> None:
        """
        Log phase-specific events within layers.

        Args:
            phase: Phase name (response, self_review, devils_advocate, synthesis)
            layer_num: Layer number
            event_type: Start/End/Error
            duration: Phase duration
            cognitive_function: Cognitive function being performed
        """
        event = {
            "event": f"Phase {event_type}",
            "phase": phase,
            "layer_number": layer_num,
            "cognitive_function": cognitive_function,
            "duration": duration
        }

        if duration and event_type == "End":
            self.performance["phase_times"][phase].append(duration)

        self.log(event)

    def log_decision_point(self,
                          decision_type: str,
                          context: Dict[str, Any],
                          outcome: str) -> None:
        """
        Log critical decision points in the workflow.

        Args:
            decision_type: Type of decision (dissent_override, confidence_adjustment, etc.)
            context: Context information for the decision
            outcome: Decision outcome
        """
        event = {
            "event": "Decision Point",
            "decision_type": decision_type,
            "outcome": outcome,
            **context
        }

        # Track dissent points
        if decision_type in ["dissent_identified", "dissent_override"]:
            self.metrics["dissent_points"].append({
                "type": decision_type,
                "context": context,
                "outcome": outcome,
                "timestamp": datetime.now().isoformat()
            })

        self.log(event)

    def log_confidence_assessment(self,
                                 phase: str,
                                 confidence_level: str,
                                 reasoning: str,
                                 layer_num: int) -> None:
        """
        Log confidence assessments throughout the workflow.

        Args:
            phase: Phase where confidence was assessed
            confidence_level: HIGH/MEDIUM/LOW
            reasoning: Reasoning for confidence level
            layer_num: Layer number
        """
        event = {
            "event": "Confidence Assessment",
            "phase": phase,
            "confidence_level": confidence_level,
            "reasoning": reasoning,
            "layer_number": layer_num
        }

        self.metrics["confidence_levels"].append({
            "phase": phase,
            "level": confidence_level,
            "reasoning": reasoning,
            "layer": layer_num,
            "timestamp": datetime.now().isoformat()
        })

        self.log(event)

    def log_resource_usage(self, resource_type: str, usage: Dict[str, Any]) -> None:
        """
        Log resource usage statistics.

        Args:
            resource_type: Type of resource (api_calls, tokens, etc.)
            usage: Usage statistics
        """
        event = {
            "event": "Resource Usage",
            "resource_type": resource_type,
            **usage
        }
        self.log(event)

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the workflow run.

        Returns:
            Dictionary with summary statistics
        """
        total_duration = time.time() - self.performance["start_time"]

        return {
            "run_id": self.run_id,
            "total_duration": total_duration,
            "total_tokens": self.performance["total_tokens"],
            "total_api_calls": sum(self.metrics["api_calls"].values()),
            "cognitive_function_distribution": dict(self.metrics["cognitive_function_calls"]),
            "model_usage": dict(self.metrics["api_calls"]),
            "layer_count": len(self.performance["layer_times"]),
            "avg_layer_time": sum(self.performance["layer_times"]) / len(self.performance["layer_times"]) if self.performance["layer_times"] else 0,
            "dissent_points_count": len(self.metrics["dissent_points"]),
            "error_count": sum(self.metrics["errors"].values())
        }

    def close(self) -> None:
        """Close the tracer and finalize logging."""
        if self.enabled and self.json_file:
            # Log final summary
            summary = self.get_summary_stats()
            self.log_workflow_event("Summary", summary)

            self.json_file.close()
            logger.info(f"Single Agent Tracer closed for run_id: {self.run_id}")