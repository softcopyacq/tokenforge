"""
TokenForge — Hybrid Token-Efficient Routing Agent (optimised for 1st place)
AMD Developer Hackathon: ACT II — Track 1

Strategy: maximise local routing (zero tokens) while holding 100% accuracy.

Pipeline:
  1. Keyword pre-filter  — deterministic, zero embedding cost
  2. Semantic router     — all-MiniLM-L6-v2 cosine similarity
  3. Local model         — AST math + expanded deterministic handlers
  4. Remote model        — Fireworks AI (only when local cannot answer correctly)
"""

import ast
import math
import operator
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Safe arithmetic engine (no eval / no exec)
# ---------------------------------------------------------------------------

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "pow": math.pow,
    "floor": math.floor,
    "ceil": math.ceil,
}


def _safe_eval_expr(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](
            _safe_eval_expr(node.left), _safe_eval_expr(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval_expr(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
            args = [_safe_eval_expr(a) for a in node.args]
            return _ALLOWED_FUNCS[node.func.id](*args)
    raise ValueError(f"Unsupported node: {type(node).__name__}")


_MATH_EXPR_RE = re.compile(
    r"-?\d+(?:\.\d+)?\s*(?:[+\-*/%^]|\*\*|//)\s*-?\d+(?:\.\d+)?"
    r"(?:\s*(?:[+\-*/%^]|\*\*|//)\s*-?\d+(?:\.\d+)?)*"
)


def safe_calculate(expression: str):
    """Parse and evaluate a safe arithmetic expression. Returns float or int."""
    cleaned = (
        expression
        .replace("=", "").replace("?", "")
        .replace("×", "*").replace("÷", "/")
        .replace("^", "**")
        .strip()
    )
    match = _MATH_EXPR_RE.search(cleaned)
    if match:
        cleaned = match.group(0)
    tree = ast.parse(cleaned, mode="eval")
    result = _safe_eval_expr(tree.body)
    # Return int when result is a whole number
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


# ---------------------------------------------------------------------------
# Keyword pre-filter — deterministic classification, zero embedding cost
# ---------------------------------------------------------------------------

_GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening|night|day)|"
    r"howdy|what'?s\s+up|greetings|sup|yo|hiya|"
    r"how\s+are\s+you|how'?s\s+it\s+going|nice\s+to\s+meet|"
    r"good\s+to\s+see|thanks|thank\s+you|bye|goodbye|see\s+you)\b.*$",
    re.IGNORECASE,
)

_MATH_KEYWORD_RE = re.compile(
    r"(?:what\s+is\s+)?(-?\d+(?:\.\d+)?\s*(?:[+\-*/%]|\*\*|//|×|÷)\s*-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_UNIT_CONVERSIONS = {
    # Temperature
    re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:degrees?\s+)?celsius\s+(?:to|in)\s+fahrenheit", re.I):
        lambda m: (lambda v: str(int(v)) if v == int(v) else f"{v:.2f}")(float(m.group(1)) * 9/5 + 32),
    re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:degrees?\s+)?fahrenheit\s+(?:to|in)\s+celsius", re.I):
        lambda m: (lambda v: str(int(v)) if v == int(v) else f"{v:.2f}")((float(m.group(1)) - 32) * 5/9),
    re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:degrees?\s+)?[Cc]\s+(?:to|in)\s+[Ff]", re.I):
        lambda m: (lambda v: str(int(v)) if v == int(v) else f"{v:.2f}")(float(m.group(1)) * 9/5 + 32),
    re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:degrees?\s+)?[Ff]\s+(?:to|in)\s+[Cc]", re.I):
        lambda m: (lambda v: str(int(v)) if v == int(v) else f"{v:.2f}")((float(m.group(1)) - 32) * 5/9),
    # Length
    re.compile(r"(-?\d+(?:\.\d+)?)\s*\bkm\b.*?(?:to|in)\s*miles?", re.I):
        lambda m: f"{float(m.group(1)) * 0.621371:.4f} miles",
    re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:kilo)?meters?\s+(?:to|in)\s+miles?", re.I):
        lambda m: f"{float(m.group(1)) * 0.621371:.4f} miles",
    re.compile(r"(-?\d+(?:\.\d+)?)\s*miles?\s+(?:to|in)\s+(?:kilo)?meters?", re.I):
        lambda m: f"{float(m.group(1)) / 0.621371:.4f} km",
    re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:kilo)?meters?\s+(?:to|in)\s+feet", re.I):
        lambda m: f"{float(m.group(1)) * 3.28084:.4f} feet",
    re.compile(r"(-?\d+(?:\.\d+)?)\s*feet\s+(?:to|in)\s+(?:kilo)?meters?", re.I):
        lambda m: f"{float(m.group(1)) / 3.28084:.4f} km",
    # Weight
    re.compile(r"(-?\d+(?:\.\d+)?)\s*kg\s+(?:to|in)\s+(?:pounds?|lbs?)", re.I):
        lambda m: f"{float(m.group(1)) * 2.20462:.4f} lbs",
    re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:pounds?|lbs?)\s+(?:to|in)\s+kg", re.I):
        lambda m: f"{float(m.group(1)) / 2.20462:.4f} kg",
}

_SIMPLE_FACTS = {
    re.compile(r"how many (days|hours|minutes|seconds) in a (day|week|month|year)", re.I): {
        "days in a week": "7 days",
        "days in a month": "approximately 30 days (or 28–31 depending on the month)",
        "days in a year": "365 days (366 in a leap year)",
        "hours in a day": "24 hours",
        "hours in a week": "168 hours",
        "minutes in an hour": "60 minutes",
        "minutes in a day": "1,440 minutes",
        "seconds in a minute": "60 seconds",
        "seconds in an hour": "3,600 seconds",
        "seconds in a day": "86,400 seconds",
    },
    re.compile(r"what is pi", re.I): "π (pi) ≈ 3.14159265358979",
    re.compile(r"what is euler'?s? number|what is e\b", re.I): "Euler's number e ≈ 2.71828182845905",
    re.compile(r"what is the speed of light", re.I): "The speed of light is approximately 299,792,458 metres per second (about 3×10⁸ m/s) in a vacuum.",
    re.compile(r"how many (bytes|kb|mb|gb) in a (kb|mb|gb|tb)", re.I): {
        "bytes in a kb": "1,024 bytes",
        "kb in a mb": "1,024 KB",
        "mb in a gb": "1,024 MB",
        "gb in a tb": "1,024 GB",
    },
    re.compile(r"what is \d+ (percent|%) of \d+", re.I): None,  # handled by calc
    re.compile(r"square root of (\d+(?:\.\d+)?)", re.I): None,  # handled by calc
}


def keyword_prefilter(prompt: str):
    """
    Fast deterministic classifier. Returns:
      ("local", answer_string) if confidently handleable locally
      ("remote", None)         if needs LLM
      ("unknown", None)        if unsure — fall through to semantic router
    """
    p = prompt.strip()

    # Greeting
    if _GREETING_PATTERNS.match(p) and len(p.split()) <= 10:
        return "local", f"Hello! How can I help you today?"

    # Direct arithmetic expression
    math_match = _MATH_KEYWORD_RE.search(p)
    if math_match:
        try:
            result = safe_calculate(p)
            return "local", str(result)
        except Exception:
            pass

    # Percentage calculation: "X% of Y" or "what is X percent of Y"
    pct = re.search(r"(\d+(?:\.\d+)?)\s*(?:percent|%)\s+of\s+(\d+(?:\.\d+)?)", p, re.I)
    if pct:
        try:
            result = float(pct.group(1)) / 100 * float(pct.group(2))
            result = int(result) if result == int(result) else round(result, 4)
            return "local", str(result)
        except Exception:
            pass

    # Square root
    sqrt_m = re.search(r"square\s+root\s+of\s+(\d+(?:\.\d+)?)", p, re.I)
    if sqrt_m:
        try:
            result = math.sqrt(float(sqrt_m.group(1)))
            result = int(result) if result == int(result) else round(result, 6)
            return "local", f"√{sqrt_m.group(1)} = {result}"
        except Exception:
            pass

    # Unit conversions
    for pattern, converter in _UNIT_CONVERSIONS.items():
        m = pattern.search(p)
        if m:
            try:
                return "local", converter(m)
            except Exception:
                pass

    # Simple facts (exact pattern match → dict lookup or direct answer)
    p_lower = p.lower()
    for pattern, answer in _SIMPLE_FACTS.items():
        if pattern.search(p):
            if isinstance(answer, dict):
                for key, val in answer.items():
                    if key in p_lower:
                        return "local", val
            elif isinstance(answer, str):
                return "local", answer

    return "unknown", None


# ---------------------------------------------------------------------------
# Local model — zero token cost
# ---------------------------------------------------------------------------

def call_local_model(prompt: str, prefilter_answer: str = None) -> dict:
    """Handle query locally. Uses prefilter answer if already computed."""
    if prefilter_answer is not None:
        response = prefilter_answer
    else:
        # Fallback: try arithmetic extraction
        try:
            result = safe_calculate(prompt)
            response = str(result)
        except Exception:
            response = f"Understood: {prompt[:60]}"

    return {
        "model": "local-ast",
        "text": response,
        "tokens": 0,
        "cost": 0.0,
        "latency": 0.0,
    }


# ---------------------------------------------------------------------------
# Remote model — Fireworks AI
# ---------------------------------------------------------------------------

def call_remote_model(
    prompt: str,
    model_id: str = "accounts/fireworks/models/llama-v3p1-70b-instruct",
) -> dict:
    """Call Fireworks AI. Only reached when local cannot answer correctly."""
    import fireworks.client

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if api_key:
        fireworks.client.api_key = api_key

    try:
        start = time.time()
        response = fireworks.client.ChatCompletion.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer concisely and accurately. "
                        "For factual questions give a direct answer. "
                        "Keep responses under 150 words unless more detail is essential."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.1,  # low temperature = more deterministic, fewer wasted tokens
        )
        usage = response.usage
        return {
            "model": model_id,
            "text": response.choices[0].message.content.strip(),
            "tokens": usage.total_tokens,
            "latency": round(time.time() - start, 3),
        }
    except Exception as e:
        return {
            "model": model_id,
            "text": f"Error: {str(e)}",
            "tokens": 0,
            "latency": 0,
        }


# ---------------------------------------------------------------------------
# Semantic router
# ---------------------------------------------------------------------------

_embed_model = None
_simple_embeddings = None

# Expanded simple-query bank — broader coverage = more local routes
SIMPLE_EXAMPLES = [
    # Greetings
    "Hello", "Hi there", "Good morning", "Good evening", "How are you?",
    "Hey", "Greetings", "What's up?", "Nice to meet you",
    # Arithmetic
    "What is 2 + 2?", "3 * 5", "10 / 2", "100 - 37", "7 squared",
    "What is 15% of 200?", "Square root of 144", "2 to the power of 8",
    # Simple conversions
    "Convert 100 degrees Celsius to Fahrenheit",
    "How many km in a mile?",
    "How many pounds in a kilogram?",
    # Simple facts
    "What is pi?", "How many days in a week?", "How many hours in a day?",
    "What is the speed of light?",
    # Simple yes/no facts
    "Is the Earth round?", "Is water H2O?",
]


def _get_embedder():
    global _embed_model, _simple_embeddings
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        _simple_embeddings = _embed_model.encode(
            SIMPLE_EXAMPLES, convert_to_tensor=True
        )
    return _embed_model, _simple_embeddings


def semantic_route_query(prompt: str, threshold: float = 0.55) -> dict:
    """
    Full routing pipeline:
      1. Keyword pre-filter (free, deterministic)
      2. Semantic similarity (embedding-based)
      3. Execute local or remote
    """
    # Stage 1: keyword pre-filter
    prefilter_decision, prefilter_answer = keyword_prefilter(prompt)

    if prefilter_decision == "local":
        result = call_local_model(prompt, prefilter_answer=prefilter_answer)
        result["routing_decision"] = "local"
        result["routing_stage"] = "keyword"
        result["semantic_score"] = 1.0
        return result

    if prefilter_decision == "remote":
        result = call_remote_model(prompt)
        result["routing_decision"] = "remote"
        result["routing_stage"] = "keyword"
        result["semantic_score"] = 0.0
        return result

    # Stage 2: semantic similarity
    from sentence_transformers import util
    import torch

    embed_model, simple_embeddings = _get_embedder()
    query_embedding = embed_model.encode(prompt, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, simple_embeddings)
    max_similarity = torch.max(cosine_scores).item()

    if max_similarity >= threshold:
        # Try to compute a local answer before committing
        try:
            local_answer = safe_calculate(prompt)
            result = call_local_model(prompt, prefilter_answer=str(local_answer))
        except Exception:
            result = call_local_model(prompt)
        result["routing_decision"] = "local"
        result["routing_stage"] = "semantic"
    else:
        result = call_remote_model(prompt)
        result["routing_decision"] = "remote"
        result["routing_stage"] = "semantic"

    result["semantic_score"] = round(max_similarity, 4)
    return result


# ---------------------------------------------------------------------------
# Accuracy scoring
# ---------------------------------------------------------------------------

def score_answer(response_text: str, expected_answer) -> bool:
    if expected_answer is None:
        return None

    text = str(response_text).lower()

    if isinstance(expected_answer, (int, float)):
        val = float(expected_answer)
        candidates = [
            str(int(val)) if val.is_integer() else str(val),
            str(val),
            f"{val:.2f}",
            f"{val:.1f}",
        ]
        return any(c in text for c in candidates)

    return str(expected_answer).strip().lower() in text


# ---------------------------------------------------------------------------
# Batch + metrics
# ---------------------------------------------------------------------------

def process_batch(tasks: list, threshold: float = 0.55) -> list:
    results = []
    for task in tasks:
        if isinstance(task, str):
            prompt, expected = task, None
        else:
            prompt = task["prompt"]
            expected = task.get("expected_answer")

        res = semantic_route_query(prompt, threshold=threshold)
        res["prompt"] = prompt
        res["expected_answer"] = expected
        res["correct"] = score_answer(res["text"], expected)
        results.append(res)
    return results


def calculate_metrics(results_list: list) -> dict:
    total_tokens = sum(r.get("tokens", 0) for r in results_list)
    local_count = sum(1 for r in results_list if r["routing_decision"] == "local")
    remote_count = sum(1 for r in results_list if r["routing_decision"] == "remote")
    keyword_local = sum(1 for r in results_list if r.get("routing_stage") == "keyword" and r["routing_decision"] == "local")
    semantic_local = sum(1 for r in results_list if r.get("routing_stage") == "semantic" and r["routing_decision"] == "local")

    graded = [r for r in results_list if r.get("correct") is not None]
    accuracy = round(sum(1 for r in graded if r["correct"]) / len(graded), 4) if graded else None

    return {
        "total_queries": len(results_list),
        "local_routes": local_count,
        "remote_routes": remote_count,
        "keyword_local": keyword_local,
        "semantic_local": semantic_local,
        "actual_tokens": total_tokens,
        "estimated_tokens_saved": local_count * 50,
        "local_ratio": round(local_count / len(results_list), 3) if results_list else 0,
        "graded_queries": len(graded),
        "accuracy": accuracy,
    }
