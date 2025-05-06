import os
import json
import time

class Tracer:
    """
    Tracer for logging API calls and responses in both markdown and JSONL formats.
    Each run is saved in a timestamped subfolder under Traces/markdown and Traces/json.
    """
    def __init__(self, run_id: str, enabled: bool = True, base_dir: str = "Traces"):
        self.enabled = enabled
        self.run_id = run_id
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.md_dir = os.path.join(base_dir, "markdown", run_id)
        self.json_dir = os.path.join(base_dir, "json", run_id)
        self.md_path = os.path.join(self.md_dir, f"trace_{self.timestamp}.md")
        self.json_path = os.path.join(self.json_dir, f"trace_{self.timestamp}.jsonl")
        if enabled:
            os.makedirs(self.md_dir, exist_ok=True)
            os.makedirs(self.json_dir, exist_ok=True)
            self.md_file = open(self.md_path, "a", encoding="utf-8")
            self.json_file = open(self.json_path, "a", encoding="utf-8")

    def log(self, event: dict):
        if not self.enabled:
            return
        event['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        # Write JSONL
        self.json_file.write(json.dumps(event) + "\n")
        self.json_file.flush()
        # Write Markdown
        self.md_file.write(self._format_md(event))
        self.md_file.flush()

    def _format_md(self, event: dict) -> str:
        lines = [f"### {event.get('event', 'API Event')} ({event['timestamp']})"]
        for k, v in event.items():
            if k not in ("event", "timestamp"):
                lines.append(f"- **{k}**: `{v}`")
        lines.append("\n")
        return "\n".join(lines)

    def close(self):
        if self.enabled:
            self.md_file.close()
            self.json_file.close()