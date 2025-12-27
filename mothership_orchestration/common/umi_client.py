import importlib
import json
import os
import sys
import time
import hashlib
from typing import Any, Dict, Optional


class UnifiedMemoryClient:
    """
    Thin adapter around your existing UnifiedMemoryInterface.
    If import fails, falls back to local JSONL file so you never lose data.
    Env overrides:
      - UMI_MODULE (default: memory_system_improvements)
      - UMI_CLASS  (default: UnifiedMemoryInterface)
      - PROJECT_ROOT (optional path to add to sys.path before import)
    """
    def __init__(self):
        self._impl = None
        self._ok = False
        self._err = None
        self._module_name = None
        self._class_name = None

        module_name = os.getenv("UMI_MODULE", "memory_system_improvements")
        class_name = os.getenv("UMI_CLASS", "UnifiedMemoryInterface")
        project_root = os.getenv("PROJECT_ROOT")

        try:
            if project_root and project_root not in sys.path:
                sys.path.insert(0, project_root)
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
            # Try to construct without args; if that fails, provide sensible defaults
            try:
                self._impl = cls()
            except TypeError:
                default_db = os.getenv("UMI_DB_PATH", "ai_learning_memory.db")
                default_mem = os.getenv("UMI_MEMORY_FILE", "agent_memory.json")
                self._impl = cls(db_path=default_db, memory_file=default_mem)
            self._ok = True
            self._module_name = module_name
            self._class_name = class_name
        except Exception as e:
            self._err = e
            self._impl = None
            self._ok = False

        # If initial import failed, attempt a best-effort auto-root detection and retry once
        if not self._ok:
            try:
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                mod = importlib.import_module(module_name)
                cls = getattr(mod, class_name)
                try:
                    self._impl = cls()
                except TypeError:
                    default_db = os.getenv("UMI_DB_PATH", "ai_learning_memory.db")
                    default_mem = os.getenv("UMI_MEMORY_FILE", "agent_memory.json")
                    self._impl = cls(db_path=default_db, memory_file=default_mem)
                self._ok = True
                self._module_name = module_name
                self._class_name = class_name
                self._err = None
            except Exception as e2:
                # Preserve original error but annotate
                self._err = self._err or e2

        self._fallback_path = os.getenv("UMI_FALLBACK_JSONL", "learnings_fallback.jsonl")

    @staticmethod
    def _ts() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _fallback_write(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(self._fallback_path) or ".", exist_ok=True)
        with open(self._fallback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return {"status": "stored_fallback_jsonl", "path": self._fallback_path, "id": payload.get("id")}

    def store_learning(
        self,
        *,
        task: str,
        solution: Dict[str, Any],
        score: float = 100.0,
        feedback: str = "",
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calls your UMI.store_learning if available; tries both dict and kwarg signatures.
        Falls back to JSONL if import/signature mismatch happens.
        """
        payload = {
            "id": solution.get("id") if isinstance(solution, dict) else None,
            "task": task,
            "success": 1 if success else 0,
            "score": float(score),
            "feedback": feedback,
            "solution_json": solution,
            "metadata": metadata or {},
            "created_at": self._ts(),
        }
        # Common hash key many stacks use to dedup
        if not payload.get("id"):
            hasher = hashlib.sha256()
            hasher.update((task + json.dumps(solution, sort_keys=True)).encode("utf-8"))
            payload["id"] = hasher.hexdigest()

        if self._ok and self._impl:
            # Try dict signature
            try:
                return self._impl.store_learning(payload)
            except TypeError:
                # Try kwargs signature
                try:
                    return self._impl.store_learning(
                        task=task,
                        solution_json=solution,
                        score=score,
                        feedback=feedback,
                        success=success,
                        metadata=metadata or {},
                    )
                except Exception:
                    pass
            except Exception:
                pass

        # Fallback JSONL
        return self._fallback_write(payload)

    # Pass-through helpers if your UMI exposes these. Silently no-op if not present.
    def share_knowledge(self, **kwargs) -> Optional[Any]:
        if self._ok and hasattr(self._impl, "share_knowledge"):
            fn = getattr(self._impl, "share_knowledge")
            try:
                # Try kwargs signature first
                return fn(**kwargs)
            except TypeError:
                # Fall back to single-dict positional argument
                return fn(kwargs)
        return None

    def retrieve_for_task(self, **kwargs) -> Optional[Any]:
        if self._ok and hasattr(self._impl, "retrieve_for_task"):
            return getattr(self._impl, "retrieve_for_task")(**kwargs)
        return None

    @property
    def status(self) -> Dict[str, Any]:
        """Diagnostic information for CLI/status checks."""
        return {
            "ok": self._ok,
            "module": self._module_name,
            "class": self._class_name,
            "fallback_path": self._fallback_path,
            "error": str(self._err) if self._err else None,
        }
