import asyncio
import contextvars
import csv
import logging
import os
import time
from pathlib import Path
from typing import Any, List, Optional

logging.basicConfig(level=logging.INFO)

from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from llm_client import LiteLLMClient

from mind2web2.utils.load_eval_script import load_eval_script
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.eval_runner import DualSemaphore
from mind2web2.utils.page_info_retrieval import BatchBrowserManager

_task_browsers: dict[str, list[BatchBrowserManager]] = {}
_original_bbm_start = BatchBrowserManager.start
_current_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("_current_task_id", default=None)

async def _patched_bbm_start(self):
    await _original_bbm_start(self)
    task_id = _current_task_id.get()
    if task_id is not None:
        _task_browsers.setdefault(task_id, []).append(self)
    logging.info(f"Browser started for task {task_id}")

BatchBrowserManager.start = _patched_bbm_start


async def _cleanup_browsers(task_id: str):
    browsers = _task_browsers.pop(task_id, [])
    for mgr in browsers:
        try:
            await mgr.stop()
            logging.info(f"Browser stopped for task {task_id}")
        except Exception as e:
            logging.warning(f"Failed to stop browser for task {task_id}: {e}")

DATA_DIR = os.getenv("DATA_DIR", "mind2web2-data")
CACHE_DIR = os.getenv("CACHE_DIR", "cache")


class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl]
    config: dict[str, Any]


def find_eval_scripts_dir(data_dir: str) -> Path | None:
    """Find the eval scripts directory under data_dir (handles nested version dirs)."""
    base = Path(data_dir) / "evaluation_scripts"
    if not base.exists():
        return None
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return base


def load_domain_task_ids(data_dir: str, domain: str) -> set[str] | None:
    """Load task IDs from a domain CSV (dev_set.csv or test_set.csv)."""
    csv_path = Path(data_dir) / f"{domain}.csv"
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return {row["task_id"] for row in reader}


def discover_tasks(scripts_dir: Path) -> dict[str, Path]:
    """Discover all eval script files in a directory, returning {task_id: path}."""
    scripts = {}
    if not scripts_dir.exists():
        return scripts
    for f in sorted(scripts_dir.iterdir()):
        if f.is_dir():
            for sub in sorted(f.iterdir()):
                if sub.suffix == ".py" and not sub.name.startswith("_"):
                    scripts[sub.stem] = sub
        elif f.suffix == ".py" and not f.name.startswith("_"):
            scripts[f.stem] = f
    return scripts


def get_task_description(script_path: Path) -> str:
    """Extract TASK_DESCRIPTION from an eval script module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "TASK_DESCRIPTION", f"Complete the task: {script_path.stem}")


class Agent:
    required_roles: list[str] = ["agent"]
    required_config_keys: list[str] = []

    def __init__(self):
        self.messenger = Messenger()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        domain = request.config.get("domain", "dev_set")
        task_ids_filter = request.config.get("task_ids", None)
        num_tasks = request.config.get("num_tasks", None)
        judge_model = request.config.get("judge_model") or os.getenv("AGENT_LLM", "openai/gpt-4o-mini")
        agent_url = str(request.participants["agent"])

        scripts_dir = find_eval_scripts_dir(DATA_DIR)
        if not scripts_dir:
            await updater.reject(
                new_agent_text_message(f"No evaluation_scripts dir found in {DATA_DIR}")
            )
            return

        all_tasks = discover_tasks(scripts_dir)

        domain_ids = load_domain_task_ids(DATA_DIR, domain)
        if domain_ids is not None:
            all_tasks = {k: v for k, v in all_tasks.items() if k in domain_ids}

        if task_ids_filter:
            all_tasks = {k: v for k, v in all_tasks.items() if k in task_ids_filter}
        if num_tasks:
            all_tasks = dict(list(all_tasks.items())[:num_tasks])

        num_instances = request.config.get("num_instances", None)
        if num_instances is not None:
            all_tasks = dict(list(all_tasks.items())[:int(num_instances)])

        shard_index = request.config.get("shard_index", None)
        num_shards = request.config.get("num_shards", None)
        if num_shards is not None and int(num_shards) > 1:
            idx = int(shard_index or 0)
            n = int(num_shards)
            items = list(all_tasks.items())
            all_tasks = dict(items[i] for i in range(len(items)) if i % n == idx)

        if not all_tasks:
            result_data = {
                "domain": domain,
                "score": 0,
                "max_score": 0,
                "avg_score": 0,
                "task_scores": {},
                "time_used": 0,
            }
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=f"No tasks assigned to this shard for domain={domain}")),
                    Part(root=DataPart(data=result_data)),
                ],
                name="Result",
            )
            return

        max_concurrent = int(request.config.get("max_concurrent", 10))

        shard_info = ""
        if num_shards is not None and int(num_shards) > 1:
            shard_info = f", shard {int(shard_index or 0)}/{int(num_shards)}"

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Evaluating {len(all_tasks)} tasks from {domain} (concurrency={max_concurrent}{shard_info})"
            )
        )

        client = LiteLLMClient()
        webpage_semaphore = asyncio.Semaphore(5)
        llm_semaphore = asyncio.Semaphore(30)
        semaphore = DualSemaphore(webpage_semaphore, llm_semaphore)
        task_concurrency = asyncio.Semaphore(max_concurrent)

        task_scores = {}
        start_time = time.time()

        async def _run_one(task_id: str, script_path: Path) -> None:
            async with task_concurrency:
                logging.info(f"Starting task {task_id}...")
                try:
                    score = await self._run_single_task(
                        task_id=task_id,
                        script_path=script_path,
                        agent_url=agent_url,
                        client=client,
                        semaphore=semaphore,
                        judge_model=judge_model,
                    )
                    task_scores[task_id] = score
                    logging.info(f"Task {task_id} scored: {score:.3f}")
                except Exception as e:
                    logging.error(f"Task {task_id} failed: {e}", exc_info=True)
                    task_scores[task_id] = 0.0

        try:
            await asyncio.gather(
                *(_run_one(tid, path) for tid, path in all_tasks.items())
            )

            time_used = time.time() - start_time
            num_completed = len(task_scores)
            total_score = sum(task_scores.values())
            avg_score = (total_score / num_completed) if num_completed > 0 else 0

            result_data = {
                "domain": domain,
                "score": total_score,
                "max_score": num_completed,
                "avg_score": avg_score,
                "task_scores": task_scores,
                "time_used": time_used,
            }

            task_results_str = "\n".join(
                f"  {tid}: {score:.3f}"
                for tid, score in task_scores.items()
            )

            summary = f"""Mind2Web-2 Benchmark Results
Domain: {domain}
Tasks: {num_completed}
Avg Score: {avg_score:.3f}
Total Score: {total_score:.3f}/{num_completed}
Time: {time_used:.1f}s

Task Results:
{task_results_str}"""

            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=summary)),
                    Part(root=DataPart(data=result_data)),
                ],
                name="Result",
            )

        finally:
            self.messenger.reset()

    async def _run_single_task(
        self,
        task_id: str,
        script_path: Path,
        agent_url: str,
        client: LiteLLMClient,
        semaphore: DualSemaphore,
        judge_model: Optional[str] = None,
    ) -> float:
        task_description = get_task_description(script_path)

        t0 = time.time()
        answer = await self.messenger.talk_to_agent(
            message=task_description,
            url=agent_url,
            new_conversation=True,
            timeout=600,
        )
        t_purple = time.time() - t0
        logging.info(f"[{task_id}] Purple agent responded in {t_purple:.1f}s ({len(answer)} chars)")
        logging.info(f"[{task_id}] Answer preview:\n{answer[:500]}...")

        t1 = time.time()
        eval_fn = load_eval_script(str(script_path))

        cache_dir = os.path.join(CACHE_DIR, task_id)
        os.makedirs(cache_dir, exist_ok=True)
        cache = CacheFileSys(task_dir=cache_dir)

        logger = logging.getLogger(f"mind2web2.eval.{task_id}")

        eval_kwargs = dict(
            client=client,
            answer=answer,
            agent_name="purple_agent",
            answer_name="answer_1",
            cache=cache,
            semaphore=semaphore,
            logger=logger,
        )
        if judge_model:
            eval_kwargs["model"] = judge_model

        _current_task_id.set(task_id)
        try:
            result = await eval_fn(**eval_kwargs)
        finally:
            await _cleanup_browsers(task_id)
        cache.save()
        t_eval = time.time() - t1

        score = float(result.get("final_score", 0.0))
        t_total = time.time() - t0
        logging.info(f"[{task_id}] Timing: purple={t_purple:.1f}s eval={t_eval:.1f}s total={t_total:.1f}s")
        _log_eval_result(task_id, result)
        return score


def _format_tree(node: dict, indent: int = 0) -> str:
    prefix = "  " * indent
    status_icon = {"passed": "+", "failed": "X", "skipped": "-", "partial": "~", "initialized": "?"}.get(node.get("status", "?"), "?")
    line = f"{prefix}[{status_icon}] {node['id']}: {node['desc']} (score={node.get('score', 0):.2f}, status={node.get('status', '?')})"
    lines = [line]
    for child in node.get("children", []):
        lines.append(_format_tree(child, indent + 1))
    return "\n".join(lines)


def _log_eval_result(task_id: str, result: dict) -> None:
    score = result.get("final_score", 0.0)
    judge = result.get("judge_model", "?")
    logging.info(f"[{task_id}] Final score: {score:.3f} (judge={judge})")

    for breakdown in result.get("eval_breakdown", []):
        tree = breakdown.get("verification_tree")
        if tree:
            logging.info(f"[{task_id}] Verification tree:\n{_format_tree(tree)}")

        info_list = breakdown.get("info", [])
        for info in info_list:
            for key, val in info.items():
                if key == "no_info":
                    continue
                logging.info(f"[{task_id}] {key}: {_truncate(val)}")


def _truncate(val, max_len=300) -> str:
    s = str(val)
    return s if len(s) <= max_len else s[:max_len] + "..."
