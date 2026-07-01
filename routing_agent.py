"""
Hybrid Token-Efficient Routing Agent
AMD Developer Hackathon: ACT II — Track 1

Routes each incoming query to either a local (zero-token) model or a
remote Fireworks AI model, based on semantic complexity, while tracking
token usage and answer accuracy.
"""

import ast
import operator
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Local model (zero-cost, runs on the standardized scoring environment)
# ---------------------------------------------------------------------------

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _safe_eval_expr(node):
    """Evaluate a restricted arithmetic AST node. Numbers and +,-,*,/ only."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](
            _safe_eval_expr(node.left), _safe_eval_expr(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval_expr(node.operand))
    raise ValueError("Unsupported expression")


_MATH_EXPR_RE = re.compile(r"-?\d+(?:\.\d+)?\s*[+\-*/]\s*-?\d+(?:\.\d+)?")


def safe_calculate(expression: str):
    """Safely evaluate a simple arithmetic expression. No eval(), no exec()."""
    cleaned = expression.replace("=", "").replace("?", "").strip()
    match = _MATH_EXPR_RE.search(cleaned)
    if match:
        cleaned = match.group(0)
    tree = ast.parse(cleaned, mode="eval")
    return _safe_eval_expr(tree.body)


def call_local_model(prompt: str) -> dict:
    """Simulates a local, zero-cost model for simple queries (math + greetings)."""
    is_math_like = any(op in prompt for op in ["+", "-", "*", "/"]) and len(prompt) < 30

    if is_math_like:
        try:
            result = safe_calculate(prompt)
            response = f"Local Model Result: {result}"
        except (ValueError, SyntaxError, ZeroDivisionError):
            response = "Local model could not parse simple math."
    else:
        response = f"Local Model: handled '{prompt[:30]}' as a simple/conversational task."

    return {
        "model": "local-simulated",
        "text": response,
        "tokens": 0,
        "cost": 0.0,
    }


# ---------------------------------------------------------------------------
# Remote model (Fireworks AI)
# ---------------------------------------------------------------------------

def call_remote_model(prompt: str, model_id: str = "accounts/fireworks/models/llama-v3-70b-instruct") -> dict:
    """Calls the Fireworks AI API for complex queries."""
    import fireworks.client

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if api_key:
        fireworks.client.api_key = api_key

    try:
        start_time = time.time()
        response = fireworks.client.ChatCompletion.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )
        usage = response.usage
        return {
            "model": model_id,
            "text": response.choices[0].message.content,
            "tokens": usage.total_tokens,
            "latency": time.time() - start_time,
        }
    except Exception as e:  # noqa: BLE001 — surface any API/auth/network error
        return {
            "model": model_id,
            "text": f"Error calling Fireworks AI: {str(e)}",
            "tokens": 0,
            "latency": 0,
        }


# ---------------------------------------------------------------------------
# Semantic router
# ---------------------------------------------------------------------------

_embed_model = None
_simple_embeddings = None

SIMPLE_EXAMPLES = [
    "Hello", "Hi there", "What is 2+2?", "Good morning",
    "How are you?", "3 * 5", "Help me with a quick greeting",
]


def _get_embedder():
    """Lazily load the embedding model so importing this module stays cheap."""
    global _embed_model, _simple_embeddings
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        _simple_embeddings = _embed_model.encode(SIMPLE_EXAMPLES, convert_to_tensor=True)
    return _embed_model, _simple_embeddings


def semantic_route_query(prompt: str, threshold: float = 0.5) -> dict:
    """Routes a query to local or remote based on semantic similarity to simple examples."""
    from sentence_transformers import util
    import torch

    embed_model, simple_embeddings = _get_embedder()
    query_embedding = embed_model.encode(prompt, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, simple_embeddings)
    max_similarity = torch.max(cosine_scores).item()

    if max_similarity > threshold:
        decision = "local"
        result = call_local_model(prompt)
    else:
        decision = "remote"
        result = call_remote_model(prompt)

    result["routing_decision"] = decision
    result["semantic_score"] = max_similarity
    return result


# ---------------------------------------------------------------------------
# Accuracy scoring
# ---------------------------------------------------------------------------

def score_answer(response_text: str, expected_answer) -> bool:
    """
    Lightweight correctness check.
    - If expected_answer is numeric, look for that number in the response.
    - Otherwise, fall back to a case-insensitive substring match.
    Replace with task-specific grading logic once kickoff tasks are revealed.
    """
    if expected_answer is None:
        return None  # nothing to grade against

    text = str(response_text)

    if isinstance(expected_answer, (int, float)):
        # Match the number with optional trailing .0
        candidates = [str(expected_answer), str(int(expected_answer))] if float(expected_answer).is_integer() else [str(expected_answer)]
        return any(c in text for c in candidates)

    return str(expected_answer).strip().lower() in text.lower()


# ---------------------------------------------------------------------------
# Batch processing + metrics
# ---------------------------------------------------------------------------

def process_batch(tasks: list, threshold: float = 0.5) -> list:
    """
    tasks: list of dicts like {"prompt": "...", "expected_answer": <optional>}
           or plain strings (no grading).
    """
    results = []
    for task in tasks:
        if isinstance(task, str):
            prompt, expected = task, None
        else:
            prompt, expected = task["prompt"], task.get("expected_answer")

        res = semantic_route_query(prompt, threshold=threshold)
        res["prompt"] = prompt
        res["expected_answer"] = expected
        res["correct"] = score_answer(res["text"], expected)
        results.append(res)
    return results


def calculate_metrics(results_list: list) -> dict:
    """Token usage, routing split, and accuracy across graded tasks."""
    total_tokens_used = sum(r.get("tokens", 0) for r in results_list)
    local_count = sum(1 for r in results_list if r["routing_decision"] == "local")
    remote_count = sum(1 for r in results_list if r["routing_decision"] == "remote")

    # Estimated cost if every query had instead gone remote (avg 50 tokens/remote call)
    potential_savings = local_count * 50

    graded = [r for r in results_list if r.get("correct") is not None]
    accuracy = (sum(1 for r in graded if r["correct"]) / len(graded)) if graded else None

    return {
        "total_queries": len(results_list),
        "local_routes": local_count,
        "remote_routes": remote_count,
        "actual_tokens": total_tokens_used,
        "estimated_tokens_saved": potential_savings,
        "graded_queries": len(graded),
        "accuracy": accuracy,
    }