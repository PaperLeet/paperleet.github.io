"""Authenticated application-to-judge protocol and strict report validation."""
import ast
from collections import Counter
import hashlib
from http.client import HTTPException
import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from paperleet.catalog import problem
from paperleet.judge.runner import normalize_layers


CURATED_PROBLEMS = frozenset(("rl/gae", "rl/ppo_clip", "rl/dpo", "rl/grpo", "rl/dr_grpo",
    "diffusion/schedules", "diffusion/param_conversion", "diffusion/ddpm_sampling",
    "diffusion/ddim", "diffusion/flow_matching",
    'rl/gspo',
    'rl/opd',
    'rl/dapo',
    'finetuning/lora',
    'finetuning/qlora',
    'diffusion/meanflow',
    'diffusion/jit',
    'diffusion/drifting',
    'diffusion/dmd',
    'diffusion/dmd2',
    'diffusion/consistency',
    'architecture/attention',
    'architecture/multihead_attention',
    'architecture/layer_norm',
    'architecture/positional_encoding',
    'architecture/positionwise_ffn',
    'optimization/adamw',
    'architecture/flash_attention',
    'representation/clip',
    'optimization/muon_split',
    'rl/glm5_offpolicy',
    'architecture/kimi_kda',
    'architecture/block_attention_residual',
    'architecture/situ_glu',
    'finetuning/opsd_alignment',
    'finetuning/opsd_clipping',
    'rl/sdpo_feedback',
    'rl/sdpo_topk',
    'representation/repa',
    'representation/rotation_trick',
    'architecture/differential_attention',
    'diffusion/block_diffusion',
    'finetuning/distillm2',
    'rl/pkpo',
    'finetuning/svdquant',
    'representation/moge_alignment',
    'diffusion/daps',
    'diffusion/diffusion_nft',
    'architecture/var_next_scale',
    'representation/var_tokenizer',
    'diffusion/fd_population',
    'diffusion/fd_loss',
    'representation/vicreg',
    'representation/sigreg',
    'representation/visreg',
    'representation/ijepa',
    'world_models/lewm_prediction',
    'world_models/lewm_planning',
    'world_models/causal_jepa',
    'diffusion/progressive_distillation'))
MAX_CODE_BYTES = 128 * 1024
MAX_REQUEST_BYTES = 256 * 1024
MAX_REPORT_BYTES = 2 * 1024 * 1024
JOB_TIMEOUT = 150


def job_timeout():
    """Allow a shorter hosting deadline without raising the execution ceiling."""
    try:
        timeout = int(os.environ.get("PAPERLEET_JUDGE_TIMEOUT_SECONDS", str(JOB_TIMEOUT)))
    except ValueError:
        raise ValueError("PAPERLEET_JUDGE_TIMEOUT_SECONDS must be an integer from 1 to 150") from None
    if not 1 <= timeout <= JOB_TIMEOUT:
        raise ValueError("PAPERLEET_JUDGE_TIMEOUT_SECONDS must be an integer from 1 to 150")
    return timeout


class JudgeUnavailable(RuntimeError):
    pass


class JudgeBusy(JudgeUnavailable):
    pass


class InvalidReport(JudgeUnavailable):
    pass


def strict_json(raw):
    def invalid(value):
        raise ValueError("Non-finite JSON value")
    return json.loads(raw, parse_constant=invalid)


def validate_job(payload):
    if not isinstance(payload, dict) or set(payload) - {"problem", "code", "layers"}:
        raise ValueError("Expected problem, code, and optional layers")
    identifier, code = payload.get("problem"), payload.get("code")
    if not isinstance(identifier, str) or identifier not in CURATED_PROBLEMS:
        raise ValueError("Hosted judging currently supports the maintained modules included in this release only")
    if not isinstance(code, str) or not code.strip() or len(code.encode("utf-8")) > MAX_CODE_BYTES:
        raise ValueError("Code must contain 1–131072 UTF-8 bytes")
    layers = payload.get("layers")
    if layers is not None and not isinstance(layers, (str, list)):
        raise ValueError("layers must be a list of T0, T1, T2, T3")
    return {"problem": identifier, "code": code, "layers": normalize_layers(layers)}


def expected_tests(entry, layers):
    """Read trusted test definitions without importing or running learner code."""
    expected = Counter()
    for path in sorted((entry["directory"] / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
                continue
            layer, count = "T0", 1
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute) and decorator.attr in ("t0", "t1", "t2", "t3"):
                    layer = decorator.attr.upper()
                if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "parametrize":
                    try:
                        count *= len(ast.literal_eval(decorator.args[1]))
                    except (ValueError, TypeError, IndexError):
                        raise JudgeUnavailable("The hosted test manifest needs to be rebuilt") from None
            if layer in layers:
                expected[(node.name, layer)] += count
    if not expected:
        raise JudgeUnavailable("The selected hosted test manifest is empty")
    return expected


def validate_report(report, job, entry=None):
    """Derive acceptance from complete, consistent rows, never an input flag."""
    entry = entry or problem(job["problem"])
    from paperleet.common.comparisons import normalize_comparison
    digest = hashlib.sha256(job["code"].encode()).hexdigest()
    if not isinstance(report, dict) or report.get("problem") != job["problem"] or report.get("layers") != job["layers"] or report.get("source_sha256") != digest:
        raise InvalidReport("Judge returned a report for a different submission")
    rows = report.get("tests")
    if not isinstance(rows, list) or len(rows) > 1000 or type(report.get("passed")) is not bool:
        raise InvalidReport("Judge returned an invalid test report")
    allowed, actual, nodeids = expected_tests(entry, job["layers"]), Counter(), set()
    clean = []
    for row in rows:
        if not isinstance(row, dict):
            raise InvalidReport("Judge returned an invalid test row")
        identifier, layer, nodeid = row.get("id"), row.get("layer"), row.get("nodeid")
        if not isinstance(identifier, str) or not isinstance(layer, str) or (identifier, layer) not in allowed or not isinstance(nodeid, str) or not nodeid or nodeid in nodeids:
            raise InvalidReport("Judge returned duplicate or unexpected tests")
        if type(row.get("passed")) is not bool or row.get("outcome") not in ("passed", "failed", "error", "skipped") or row["passed"] != (row["outcome"] == "passed"):
            raise InvalidReport("Judge returned inconsistent test outcomes")
        duration = row.get("duration_s", 0)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise InvalidReport("Judge returned an invalid test duration")
        if not isinstance(row.get("artifacts", {}), dict):
            raise InvalidReport("Judge returned invalid test artifacts")
        if not all(isinstance(row.get(key, ""), str) for key in ("message", "why")):
            raise InvalidReport("Judge returned invalid test details")
        actual[(identifier, layer)] += 1
        nodeids.add(nodeid)
        clean.append({"id": identifier, "nodeid": nodeid, "layer": layer, "passed": row["passed"],
            "outcome": row["outcome"], "duration_s": duration, "artifacts": row.get("artifacts", {}),
            "message": row.get("message", "")[:16000], "why": row.get("why", "")[:2000]})
        comparison = normalize_comparison(row.get("comparison")) if not row["passed"] else None
        if comparison is not None: clean[-1]["comparison"] = comparison
    if any(count > allowed[key] for key, count in actual.items()):
        raise InvalidReport("Judge returned unexpected repeated tests")
    error = report.get("error")
    if error is not None and not isinstance(error, str):
        raise InvalidReport("Judge returned an invalid error")
    complete = actual == allowed
    passed = bool(clean) and complete and all(row["passed"] for row in clean) and not error
    if report["passed"] != passed:
        raise InvalidReport("Judge returned an inconsistent completion status")
    if not complete and not error:
        raise InvalidReport("Judge did not complete every selected test")
    result = {"problem": job["problem"], "impl": "submission.py", "source_sha256": digest,
        "created_at": report.get("created_at", ""), "layers": job["layers"], "tests": clean,
        "passed": passed, "mutation": None,
        "summary": {layer: [sum(row["passed"] for row in clean if row["layer"] == layer), sum(row["layer"] == layer for row in clean)] for layer in ("T0", "T1", "T2", "T3")}}
    if error:
        result["error"] = error[:20000]
    return result


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _configuration():
    url = os.environ.get("PAPERLEET_JUDGE_URL", "").rstrip("/")
    token = os.environ.get("PAPERLEET_JUDGE_TOKEN", "")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or len(token) < 32:
        raise JudgeUnavailable("The isolated judge is not configured")
    return url, token


def _request(endpoint, payload=None, timeout=5):
    url, token = _configuration()
    request = Request(url + endpoint, data=json.dumps(payload, allow_nan=False).encode() if payload is not None else None,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        method="POST" if payload is not None else "GET")
    try:
        # Internal service traffic must not inherit an external HTTP proxy or
        # follow a redirect that could forward the shared service credential.
        with build_opener(ProxyHandler({}), _NoRedirect()).open(request, timeout=timeout) as response:
            raw = response.read(MAX_REPORT_BYTES + 1)
            if len(raw) > MAX_REPORT_BYTES:
                raise InvalidReport("Judge response exceeded the output limit")
            return strict_json(raw)
    except HTTPError as exc:
        if exc.code in (409, 429):
            raise JudgeBusy("The judge is busy. Please try again shortly.") from None
        raise JudgeUnavailable("The isolated judge is unavailable. Please try again shortly.") from None
    except (URLError, TimeoutError, OSError, ValueError, HTTPException):
        raise JudgeUnavailable("The isolated judge returned no valid response. Please try again shortly.") from None


def run_hosted_problem(entry, code, layers=None):
    job = validate_job({"problem": entry["id"], "code": code, "layers": layers})
    return validate_report(_request("/run", job, timeout=job_timeout() + 20), job, entry)


def judge_capabilities():
    try:
        response = _request("/health", timeout=6)
        if isinstance(response, dict) and response.get("ready") is True:
            return {"ready": True, "mode": "isolated", "supported_problems": sorted(CURATED_PROBLEMS)}
        return {"ready": False, "reason": "The isolated judge worker is not ready."}
    except JudgeUnavailable as exc:
        return {"ready": False, "reason": str(exc)}
