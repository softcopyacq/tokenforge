# Hybrid Token-Efficient Routing Agent

AMD Developer Hackathon: ACT II — Track 1 (Hybrid Token-Efficient Routing Agent)

## Overview
This project implements a hybrid LLM gateway that routes each incoming query to either a **local, zero-token model** or a **remote Fireworks AI model**, based on semantic similarity to known "simple" query patterns. The goal is to minimize total token usage while keeping answer accuracy above threshold.

## Architecture
- **Local model** (`call_local_model`): handles short arithmetic and greeting/conversational queries at zero token cost, using a restricted AST-based parser (no `eval`/`exec`) for safety.
- **Remote model** (`call_remote_model`): calls Fireworks AI (`llama-v3-70b-instruct` by default) for queries that require real reasoning or generation.
- **Semantic router** (`semantic_route_query`): uses `sentence-transformers` (`all-MiniLM-L6-v2`) to embed each query and compare it against a bank of known-simple example queries. If similarity exceeds a threshold, the query is routed locally; otherwise it goes to the remote model.
- **Accuracy scoring** (`score_answer`): grades each response against an optional `expected_answer` per task, supporting both numeric and substring matching.
- **Metrics** (`calculate_metrics`): reports token usage, routing split, estimated token savings, and accuracy across graded tasks.

## Project Structure
```
.
├── main.py            # CLI entrypoint — runs the agent over a task list
├── routing_agent.py    # Core routing, model, and scoring logic
├── tasks.json           # Sample task list (prompt + expected_answer pairs)
├── requirements.txt
├── Dockerfile
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Configuration
Set your Fireworks AI API key as an environment variable (never hardcode it):
```bash
export FIREWORKS_API_KEY="your_api_key_here"
```
Or place it in a `.env` file in the project root (already supported via `python-dotenv`):
```
FIREWORKS_API_KEY=your_api_key_here
```

## Usage

### Run with the built-in sample tasks
```bash
python main.py
```

### Run with a custom task list
```bash
python main.py --tasks tasks.json --output results.json
```

### Use the router directly in Python
```python
from routing_agent import semantic_route_query

result = semantic_route_query("Hello!")
print(result["routing_decision"], result["text"])
```

## Task File Format
```json
[
  {"prompt": "What is 12 * 4?", "expected_answer": 48},
  {"prompt": "Explain quantum entanglement.", "expected_answer": null}
]
```
`expected_answer` is optional — omit or set to `null` for ungraded prompts.

## Running with Docker
```bash
docker build -t routing-agent .
docker run --rm -e FIREWORKS_API_KEY="your_api_key_here" routing-agent
```
The container pre-downloads the embedding model at build time, so scoring runs do not depend on network access to Hugging Face at runtime.

## Notes on the Standardized Scoring Environment
- All tokens used by the local model count as zero toward the final score.
- The local model is intentionally lightweight (no large weights, restricted arithmetic parser) so it stays within constraints of the standardized scoring environment.
- `score_answer` uses simple heuristic matching (numeric containment / substring match) as a placeholder; once the official kickoff tasks and grading scheme are revealed, swap in the exact grading logic required.