import asyncio
import hashlib
import inspect
import time
from typing import Optional, Any, Dict, List, Callable
import logging
import yaml
from pathlib import Path
from dataclasses import dataclass

from core.domain.context import RequestContext
from core.ports.tools import ToolPort


@dataclass
class ToolDefinition:
    name: str
    module: Optional[str] = None
    function: Optional[str] = None
    handler: Optional[Callable] = None
    acl: List[str] = None
    timeout_seconds: int = 30
    idempotent: bool = False
    schema: Dict = None


class ToolPortAdapter:
    def __init__(
        self,
        config_path: str = "config/tools.yml",
        enable_audit: bool = True
    ):
        self._config_path = config_path
        self._enable_audit = enable_audit
        self._logger = logging.getLogger(__name__)
        
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable] = {}
        self._audit_log: List[Dict] = []
        
        self._load_config()
        self._load_builtin_tools()
    
    def _load_config(self) -> None:
        path = Path(self._config_path)
        if not path.exists():
            self._logger.warning(f"Tools config not found: {self._config_path}")
            return
        
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        for tool_data in config.get("tools", []):
            name = tool_data.get("name")
            if not name:
                continue
            
            self._tools[name] = ToolDefinition(
                name=name,
                module=tool_data.get("module"),
                acl=tool_data.get("acl", ["user"]),
                timeout_seconds=tool_data.get("timeout_seconds", 30),
                idempotent=tool_data.get("idempotent", False),
                schema=tool_data.get("schema")
            )
    
    def _load_builtin_tools(self) -> None:
        from agent_platform.tools.builtins import json_io, skill_echo

        builtin_handlers = {
            "skill_echo": skill_echo.skill_echo,
            "read_json_all_title": json_io.read_json_all_title,
            "read_json_context_by_title": json_io.read_json_context_by_title,
            "save_result_2_json": json_io.save_result_2_json,
        }
        for name, handler in builtin_handlers.items():
            self._handlers[name] = handler
            if name not in self._tools:
                acl = (
                    ["user", "cli", "test"]
                    if name == "skill_echo"
                    else ["reader", "writer", "user", "cli", "test"]
                )
                self._tools[name] = ToolDefinition(
                    name=name,
                    acl=acl,
                    timeout_seconds=30 if name != "skill_echo" else 10,
                )

        try:
            from agent_platform.tools.builtins import word_io

            self._handlers["read_word_2_json"] = word_io.read_word_2_json
            if "read_word_2_json" not in self._tools:
                self._tools["read_word_2_json"] = ToolDefinition(
                    name="read_word_2_json",
                    acl=["reader", "user", "cli", "test"],
                    timeout_seconds=60,
                )
        except ImportError:
            pass
    
    def register_tool(
        self,
        name: str,
        handler: Callable,
        acl: Optional[List[str]] = None,
        timeout_seconds: int = 30,
        idempotent: bool = False
    ) -> None:
        self._handlers[name] = handler
        self._tools[name] = ToolDefinition(
            name=name,
            handler=handler,
            acl=acl or ["user"],
            timeout_seconds=timeout_seconds,
            idempotent=idempotent
        )
        self._logger.info(f"Registered tool: {name}")
    
    def _check_acl(self, tool_name: str, context: RequestContext) -> bool:
        tool_def = self._tools.get(tool_name)
        if not tool_def:
            return False
        
        if tool_def.acl is None or "user" in tool_def.acl:
            return True
        
        if context.acl and hasattr(context.acl, 'tool_names'):
            return tool_name in context.acl.tool_names
        
        return True
    
    def _hash_args(self, args: Dict) -> str:
        import json
        args_str = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(args_str.encode()).hexdigest()[:16]
    
    def _audit(
        self,
        tool_name: str,
        args: Dict,
        result: Any,
        status: str,
        latency_ms: float,
        context: RequestContext
    ) -> None:
        if not self._enable_audit:
            return
        
        record = {
            "timestamp": time.time(),
            "trace_id": context.trace_id,
            "session_id": context.session_id,
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "tool_name": tool_name,
            "args_hash": self._hash_args(args),
            "status": status,
            "latency_ms": latency_ms
        }
        
        self._audit_log.append(record)
        
        self._logger.info(
            f"Tool call: {tool_name} | status={status} | latency={latency_ms:.2f}ms | "
            f"trace={context.trace_id}"
        )
    
    async def invoke(
        self,
        tool_name: str,
        args: Dict,
        context: RequestContext
    ) -> Any:
        if tool_name not in self._tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        if not self._check_acl(tool_name, context):
            raise PermissionError(f"ACL denied for tool: {tool_name}")
        
        handler = self._handlers.get(tool_name)
        if not handler:
            tool_def = self._tools[tool_name]
            if tool_def.module:
                handler = self._load_handler_from_module(tool_def.module, tool_name)
                if handler:
                    self._handlers[tool_name] = handler
        
        if not handler:
            raise RuntimeError(f"No handler for tool: {tool_name}")
        
        start_time = time.time()
        
        try:
            call_args = dict(args)
            if "context" in inspect.signature(handler).parameters:
                call_args["context"] = context
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**call_args)
            else:
                result = handler(**call_args)
            
            latency_ms = (time.time() - start_time) * 1000
            self._audit(tool_name, args, result, "success", latency_ms, context)
            
            return result
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._audit(tool_name, args, None, f"error: {e}", latency_ms, context)
            raise
    
    def _load_handler_from_module(self, module_path: str, tool_name: str) -> Optional[Callable]:
        try:
            import importlib
            module = importlib.import_module(module_path)
            func_name = tool_name.split(".")[-1]
            return getattr(module, func_name, None)
        except Exception as e:
            self._logger.error(f"Failed to load module {module_path}: {e}")
            return None
    
    async def invoke_batch(
        self,
        tool_name: str,
        args_list: List[Dict],
        context: RequestContext
    ) -> List[Any]:
        results = []
        for args in args_list:
            try:
                result = await self.invoke(tool_name, args, context)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        return results
    
    def list_tools(self) -> List[str]:
        return list(self._tools.keys())
    
    def get_schema(self, tool_name: str) -> Optional[Dict]:
        tool_def = self._tools.get(tool_name)
        return tool_def.schema if tool_def else None
    
    def validate_args(self, tool_name: str, args: Dict) -> bool:
        schema = self.get_schema(tool_name)
        if not schema:
            return True
        
        try:
            import jsonschema
            jsonschema.validate(args, schema)
            return True
        except Exception:
            return False
    
    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        return self._audit_log[-limit:]
    
    def health(self) -> dict:
        return {
            "status": "healthy",
            "tools_registered": len(self._tools),
            "audit_enabled": self._enable_audit
        }
