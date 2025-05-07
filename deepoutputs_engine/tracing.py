import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class Tracer:
    """
    Enhanced tracer for logging API calls, workflow events, and performance metrics.
    Each run is saved in a timestamped subfolder under Traces/markdown and Traces/json.
    """
    def __init__(self, run_id: str, enabled: bool = True, base_dir: str = "Traces"):
        self.enabled = enabled
        self.run_id = run_id
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.base_dir = Path(base_dir)
        self.md_dir = self.base_dir / "markdown" / run_id
        self.json_dir = self.base_dir / "json" / run_id
        self.md_path = self.md_dir / f"trace_{self.timestamp}.md"
        self.json_path = self.json_dir / f"trace_{self.timestamp}.jsonl"
        self.md_file = None
        self.json_file = None
        
        # Initialize metrics tracking
        self.metrics = {
            "api_calls": defaultdict(int),
            "token_usage": defaultdict(int),
            "latency": defaultdict(list),
            "errors": defaultdict(int)
        }
        
        # Initialize performance tracking
        self.performance = {
            "start_time": time.time(),
            "layer_times": [],
            "total_tokens": 0,
            "decision_points": defaultdict(int)
        }
        
        if enabled:
            try:
                self.md_dir.mkdir(parents=True, exist_ok=True)
                self.json_dir.mkdir(parents=True, exist_ok=True)
                self.md_file = open(self.md_path, "a", encoding="utf-8")
                self.json_file = open(self.json_path, "a", encoding="utf-8")
                logger.info(f"Tracer initialized with run_id: {run_id}")
                logger.info(f"Trace files: {self.md_path}, {self.json_path}")
            except Exception as e:
                logger.error(f"Failed to initialize tracer: {str(e)}")
                self.enabled = False
                raise

    def log(self, event: Dict[str, Any], level: str = "info") -> None:
        """
        Log an event to both markdown and JSONL files.
        
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
            
            # Write Markdown
            self.md_file.write(self._format_md(event))
            self.md_file.flush()
            
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

    def _format_md(self, event: Dict[str, Any]) -> str:
        """Format an event as markdown."""
        lines = [
            f"### {event.get('event', 'API Event')} ({event['timestamp']})",
            f"**Level:** {event.get('level', 'info')}",
            f"**Run ID:** {event.get('run_id', 'unknown')}"
        ]
        
        # Add event-specific fields
        for k, v in event.items():
            if k not in ("event", "timestamp", "level", "run_id"):
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, indent=2, ensure_ascii=False)
                lines.append(f"- **{k}**: `{v}`")
                
        lines.append("\n")
        return "\n".join(lines)

    def log_api_call(self, 
                    model: str, 
                    prompt: str, 
                    response: str, 
                    duration: float,
                    tokens_used: Optional[int] = None,
                    rate_limit_remaining: Optional[int] = None,
                    error: Optional[str] = None) -> None:
        """
        Log an API call with standardized format.
        
        Args:
            model: Name of the model used
            prompt: Input prompt
            response: Model response
            duration: Call duration in seconds
            tokens_used: Number of tokens used
            rate_limit_remaining: Remaining rate limit
            error: Error message if any
        """
        # Update metrics
        self.metrics["api_calls"][model] += 1
        if tokens_used:
            self.metrics["token_usage"][model] += tokens_used
            self.performance["total_tokens"] += tokens_used
        self.metrics["latency"][model].append(duration)
        if error:
            self.metrics["errors"][model] += 1

        event = {
            "event": "API Call",
            "model": model,
            "prompt": prompt,
            "response": response,
            "duration": duration,
            "tokens_used": tokens_used,
            "rate_limit_remaining": rate_limit_remaining,
            "error": error
        }
        self.log(event, level="error" if error else "info")

    def log_workflow_event(self, 
                         event_name: str, 
                         details: Dict[str, Any],
                         error: Optional[str] = None) -> None:
        """
        Log a workflow event with standardized format.
        
        Args:
            event_name: Name of the workflow event
            details: Event details
            error: Error message if any
        """
        event = {
            "event": f"Workflow: {event_name}",
            "details": details,
            "error": error
        }
        self.log(event, level="error" if error else "info")

    def log_layer_event(self,
                       layer_num: int,
                       event_type: str,
                       prompt: str,
                       metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a layer-related event.
        
        Args:
            layer_num: Layer number
            event_type: Type of event (start/end)
            prompt: Layer prompt
            metrics: Additional metrics
        """
        event = {
            "event": f"Layer {event_type}",
            "layer": layer_num,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "metrics": metrics or {}
        }
        self.log(event)

    def log_agent_interaction(self,
                            agent1: str,
                            agent2: str,
                            interaction_type: str,
                            details: Dict[str, Any]) -> None:
        """
        Log interactions between agents.
        
        Args:
            agent1: First agent name
            agent2: Second agent name
            interaction_type: Type of interaction
            details: Interaction details
        """
        event = {
            "event": "Agent Interaction",
            "type": interaction_type,
            "agents": [agent1, agent2],
            "context": details
        }
        self.log(event)

    def log_decision_point(self,
                          decision_type: str,
                          context: Dict[str, Any],
                          outcome: str) -> None:
        """
        Log decision points in the workflow.
        
        Args:
            decision_type: Type of decision
            context: Decision context
            outcome: Decision outcome
        """
        self.performance["decision_points"][decision_type] += 1
        
        event = {
            "event": "Decision Point",
            "type": decision_type,
            "context": context,
            "outcome": outcome
        }
        self.log(event)

    def log_resource_usage(self,
                          resource_type: str,
                          usage: Dict[str, Any]) -> None:
        """
        Log resource usage.
        
        Args:
            resource_type: Type of resource
            usage: Usage metrics
        """
        event = {
            "event": "Resource Usage",
            "type": resource_type,
            "metrics": usage
        }
        self.log(event)

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the run.
        
        Returns:
            Dictionary containing run summary
        """
        duration = time.time() - self.performance["start_time"]
        
        # Calculate average latencies
        avg_latencies = {
            model: sum(times) / len(times) if times else 0
            for model, times in self.metrics["latency"].items()
        }
        
        # Calculate error rates
        error_rates = {
            model: self.metrics["errors"][model] / self.metrics["api_calls"][model]
            for model in self.metrics["api_calls"]
            if self.metrics["api_calls"][model] > 0
        }
        
        summary = {
            "event": "Run Summary",
            "duration_seconds": duration,
            "total_tokens": self.performance["total_tokens"],
            "api_calls": dict(self.metrics["api_calls"]),
            "token_usage": dict(self.metrics["token_usage"]),
            "average_latencies": avg_latencies,
            "error_rates": error_rates,
            "decision_points": dict(self.performance["decision_points"]),
            "layer_times": self.performance["layer_times"]
        }
        
        # Log the summary
        self.log(summary)
        return summary

    def close(self) -> None:
        """Close the trace files and log final summary."""
        if self.enabled:
            try:
                # Log final summary
                self.get_summary()
                
                if self.md_file:
                    self.md_file.close()
                if self.json_file:
                    self.json_file.close()
                logger.info("Tracer closed successfully")
            except Exception as e:
                logger.error(f"Error closing tracer: {str(e)}")
                raise