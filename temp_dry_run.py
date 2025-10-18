import json
import os
from deepoutputs_engine.reports import generate_markdown_report, generate_detailed_markdown_report
from deepoutputs_engine.agents.mixture import MixtureOfAgents

def parse_trace_file(trace_path):
    """Parse the JSONL trace file and extract workflow data."""
    events = []
    with open(trace_path, 'r', encoding='utf-8') as f:
        for line in f:
            events.append(json.loads(line.strip()))

    # Extract configuration
    config_event = next(e for e in events if e['event'] == 'Workflow: Configuration')
    config = config_event['details']

    # Extract layer outputs
    layer_outputs = []
    layer_events = [e for e in events if e['event'].startswith('Layer')]
    current_layer = None
    for event in events:
        if event['event'] == 'Layer Start':
            current_layer = {
                'layer_number': event['layer'],
                'layer_prompt_details': event['prompt'],
                'initial_responses': [],
                'aggregation_responses': [],
                'synthesis': '',
                'devils_advocate': ''
            }
        elif event['event'] == 'API Call' and current_layer:
            # Extract responses from API calls
            if 'Agent' in event['model'] or 'switchpoint' in event['model']:
                if len(current_layer['initial_responses']) < len(config['models']):
                    current_layer['initial_responses'].append((f"Agent {len(current_layer['initial_responses'])+1}", event['response']))
                elif len(current_layer['aggregation_responses']) < len(config['models']):
                    current_layer['aggregation_responses'].append((f"Agent {len(current_layer['aggregation_responses'])+1}", event['response']))
            elif 'Synthesis' in event['model']:
                current_layer['synthesis'] = event['response']
            elif 'Devil' in event['model']:
                current_layer['devils_advocate'] = event['response']
        elif event['event'] == 'Layer End' and current_layer:
            layer_outputs.append(current_layer)
            current_layer = None

    # Extract final output
    final_event = next(e for e in events if e['event'] == 'API Call' and 'openrouter/auto' in e['model'])
    final_output = final_event['response']

    # Extract utilization
    utilization_event = next(e for e in events if e['event'] == 'Workflow: Agent Utilization')
    utilization = utilization_event['details']['metrics']

    # Extract agent names
    final_agent_name = "Final Agent"
    synthesis_agent_name = "Synthesis Agent"
    devils_advocate_agent_name = "Devil's Advocate Agent"

    # Create mock moa object
    moa = MixtureOfAgents(config['models'], config['num_layers'], config['include_deep_research'])

    return {
        'prompt': config['prompt'],
        'layer_details': layer_outputs,
        'final_response': final_output,
        'utilization': utilization,
        'final_agent_name': final_agent_name,
        'synthesis_agent_name': synthesis_agent_name,
        'devils_advocate_agent_name': devils_advocate_agent_name,
        'moa': moa
    }

def main():
    trace_path = r"Traces\json\prompt_convert_the_strategic_framework_20250922-204341\trace_20250922-204341.jsonl"
    data = parse_trace_file(trace_path)

    # Generate reports
    markdown_report = generate_markdown_report(
        data['prompt'],
        data['layer_details'],
        data['final_response'],
        data['utilization'],
        data['final_agent_name'],
        data['moa']
    )
    detailed_report = generate_detailed_markdown_report(
        data['prompt'],
        data['layer_details'],
        data['final_response'],
        data['utilization'],
        data['final_agent_name'],
        data['synthesis_agent_name'],
        data['devils_advocate_agent_name'],
        data['moa']
    )

    # Write to temp files
    with open('temp_markdown_report.md', 'w', encoding='utf-8') as f:
        f.write(markdown_report)

    with open('temp_detailed_report.md', 'w', encoding='utf-8') as f:
        f.write(detailed_report)

    print("Reports generated successfully!")
    print("Files: temp_markdown_report.md, temp_detailed_report.md")

if __name__ == '__main__':
    main()