# Single Agent Implementation Plan

## Implementation Checklist

### Phase 1: Infrastructure Setup
- [ ] Create complete directory structure
- [ ] Add environment variables to existing .env file
- [ ] Create configuration management system
- [ ] Set up logging and basic utilities

### Phase 2: Core Agent Classes
- [ ] Implement SingleAgent base class
- [ ] Create OpenRouterSingleAgent implementation
- [ ] Add specialized agent roles (Response, Devils Advocate, Synthesis, Final)
- [ ] Implement error handling and retry logic

### Phase 3: Workflow Orchestration
- [ ] Create SingleAgentWorkflow orchestrator class
- [ ] Implement phase management (Response → Review → Devils Advocate → Synthesis)
- [ ] Add layered processing capability
- [ ] Integrate context management across phases

### Phase 4: Prompt Engineering
- [ ] Create response agent prompts (initial + self-review)
- [ ] Develop devils advocate challenge prompts
- [ ] Build synthesis and integration prompts
- [ ] Design final agent decision prompts
- [ ] Add layer-aware prompt building

### Phase 5: Tracing & Observability
- [ ] Adapt MOA tracing system for single agent
- [ ] Track cognitive function transitions
- [ ] Measure phase performance and token usage
- [ ] Log decision points and dissent tracking

### Phase 6: Reporting System
- [ ] Adapt MOA reporting for single agent workflow
- [ ] Create phase-by-phase analysis reports
- [ ] Add cognitive function utilization metrics
- [ ] Generate dissent and uncertainty tracking

### Phase 7: Main Entry Point
- [ ] Create main.py for single agent workflow
- [ ] Add command-line interface
- [ ] Integrate with existing project structure
- [ ] Add dry-run capabilities

### Phase 8: Testing & Validation
- [ ] Create test scenarios for each workflow phase
- [ ] Validate against MOA quality benchmarks
- [ ] Performance testing and optimization
- [ ] Edge case handling verification

## File Structure Implementation Order

### 1. Directory Creation
```
./single_agent/
├── single_agent_engine/
├── reports/
├── traces/
├── prompts/
└── dry_runs/
    ├── reports/
    └── traces/
```

### 2. Core Module Files (in order)
1. `single_agent_engine/__init__.py`
2. `single_agent_engine/config.py`
3. `single_agent_engine/utils.py`
4. `single_agent_engine/agents/__init__.py`
5. `single_agent_engine/agents/base.py`
6. `single_agent_engine/agents/openrouter.py`
7. `single_agent_engine/agents/single_agent.py`

### 3. Workflow Components
1. `single_agent_engine/prompts.py`
2. `single_agent_engine/tracing.py`
3. `single_agent_engine/workflow.py`
4. `single_agent_engine/reports.py`
5. `single_agent_engine/main.py`

## Environment Variables to Add

```env
# === SINGLE AGENT WORKFLOW CONFIGURATION ===

# Single Agent Models
SA_RESPONSE_AGENT_MODEL_SYMBOLIC="anthropic/claude-3.5-sonnet"
SA_DEVILS_ADVOCATE_AGENT_MODEL_SYMBOLIC="openai/gpt-4o"
SA_SYNTHESIS_AGENT_MODEL_SYMBOLIC="google/gemini-2.0-flash-001"
SA_FINAL_AGENT_MODEL_SYMBOLIC="anthropic/claude-3.5-haiku"

# Model Indirection (like MOA pattern)
SA_RESPONSE_AGENT_MODEL="SA_RESPONSE_AGENT_MODEL_SYMBOLIC"
SA_DEVILS_ADVOCATE_AGENT_MODEL="SA_DEVILS_ADVOCATE_AGENT_MODEL_SYMBOLIC"
SA_SYNTHESIS_AGENT_MODEL="SA_SYNTHESIS_AGENT_MODEL_SYMBOLIC"
SA_FINAL_AGENT_MODEL="SA_FINAL_AGENT_MODEL_SYMBOLIC"

# Workflow Configuration
SA_NUM_LAYERS=2
SA_ENABLE_SELF_REVIEW=true
SA_OUTPUT_DIR="single_agent/reports"
SA_TRACE_DIR="single_agent/traces"
SA_DRY_RUN_DIR="single_agent/dry_runs"

# Token Limits per Phase
SA_RESPONSE_MAX_TOKENS=8000
SA_REVIEW_MAX_TOKENS=4000
SA_DEVILS_ADVOCATE_MAX_TOKENS=6000
SA_SYNTHESIS_MAX_TOKENS=8000
SA_FINAL_MAX_TOKENS=6000

# Performance Settings
SA_API_TIMEOUT=120.0
SA_API_RETRY_ATTEMPTS=3
SA_API_INITIAL_BACKOFF=1.0
```

## Key Implementation Details

### Workflow State Management
- Track current phase (Response, Review, Devils Advocate, Synthesis, Final)
- Maintain context accumulation across phases
- Preserve dissenting viewpoints throughout workflow
- Handle layer transitions and context building

### Cognitive Function Specialization
- **Response Agent**: Deep analysis with assumption tagging
- **Devils Advocate**: Aggressive challenge with contrarian solutions
- **Synthesis Agent**: Integration with explicit dissent sections
- **Final Agent**: Authoritative decision with uncertainty assessment

### Quality Assurance Mechanisms
- Self-review phase for response validation
- Mandatory devils advocate challenge
- Dissent preservation and surfacing
- Confidence level tracking throughout

### Performance Optimization
- Async execution where possible
- Token usage optimization per phase
- Caching for repeated operations
- Graceful error handling and recovery

## Integration with Existing MOA System

### Shared Components
- Reuse OpenRouter client configuration
- Leverage existing tracing infrastructure
- Adapt reporting templates
- Use common utility functions

### Differentiation Points
- Single model per cognitive function vs. multiple agents
- Sequential phase execution vs. parallel agent calls
- Specialized prompts for cognitive functions
- Different metrics and utilization tracking

## Success Criteria

### Functional Requirements
- [ ] Complete workflow execution from prompt to final output
- [ ] All phases execute successfully with proper context passing
- [ ] Dissenting viewpoints preserved and surfaced
- [ ] Configurable layer processing
- [ ] Comprehensive tracing and reporting

### Quality Requirements
- [ ] Output quality comparable to MOA system
- [ ] Proper dissent management and surfacing
- [ ] Clear cognitive function differentiation
- [ ] Robust error handling and recovery
- [ ] Performance within acceptable bounds

### Integration Requirements
- [ ] Seamless integration with existing project structure
- [ ] Compatible with existing configuration patterns
- [ ] Reuses existing infrastructure where appropriate
- [ ] Maintains code quality and documentation standards

## Next Steps

1. **Switch to Code Mode** for implementation
2. **Create directory structure** and basic files
3. **Implement core classes** starting with configuration
4. **Build workflow orchestrator** with phase management
5. **Test with sample prompts** to validate workflow
6. **Generate reports** and validate output quality
7. **Create documentation** and usage examples

This implementation plan provides a clear roadmap for creating a robust single agent workflow system that maintains the quality and rigor of the MOA approach while providing the simplicity and focus of a single-agent paradigm.