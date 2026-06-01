import asyncio
import subprocess
import json
from typing import Optional, List, Any, Dict
from dataclasses import dataclass, field
import yaml
from pathlib import Path
import time
import logging
from enum import Enum

from core.ports.mcp import (
    MCPPort,
    MCPServerConfig,
    MCPTransport,
    MCPToolInfo,
    MCPToolResult
)


class MCPConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPConnection:
    server_id: str
    state: MCPConnectionState = MCPConnectionState.DISCONNECTED
    connected_at: Optional[float] = None
    last_used_at: Optional[float] = None
    error_count: int = 0
    process: Optional[subprocess.Popen] = None


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failures = 0
        self._last_failure_time = 0
        self._state = "closed"
    
    def is_open(self) -> bool:
        if self._state == "open":
            if time.time() - self._last_failure_time > self._recovery_timeout:
                self._state = "half_open"
                return False
            return True
        return False
    
    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"
    
    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self._failure_threshold:
            self._state = "open"
    
    @property
    def state(self) -> str:
        return self._state


class EnterpriseMCPAdapter:
    def __init__(
        self,
        config_path: str = "config/mcp_servers.yml",
        max_connections_per_server: int = 3,
        default_timeout: float = 30.0,
        health_check_interval: float = 30.0,
        circuit_breaker_threshold: int = 5
    ):
        self._servers: Dict[str, MCPServerConfig] = {}
        self._connections: Dict[str, MCPConnection] = {}
        self._connection_pools: Dict[str, asyncio.Queue] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._tool_cache: Dict[str, List[MCPToolInfo]] = {}
        
        self._max_connections = max_connections_per_server
        self._default_timeout = default_timeout
        self._health_check_interval = health_check_interval
        
        self._logger = logging.getLogger(__name__)
        self._health_check_task: Optional[asyncio.Task] = None
        
        if Path(config_path).exists():
            self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        for server in config.get("mcp_servers", []):
            server_id = server.get("id")
            if not server_id:
                continue
            
            transport_str = server.get("transport", "stdio")
            transport = MCPTransport(transport_str)
            
            self._servers[server_id] = MCPServerConfig(
                server_id=server_id,
                transport=transport,
                command=server.get("command"),
                args=server.get("args", []),
                url=server.get("url"),
                acl=server.get("acl", [])
            )
            
            self._circuit_breakers[server_id] = CircuitBreaker(
                failure_threshold=circuit_breaker_threshold
            )
            
            self._connections[server_id] = MCPConnection(server_id=server_id)
            self._connection_pools[server_id] = asyncio.Queue(maxsize=self._max_connections)
    
    async def _start_health_check(self) -> None:
        if self._health_check_task:
            return
        
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                for server_id in self._servers:
                    healthy = await self._check_server_health(server_id)
                    
                    if not healthy:
                        self._logger.warning(f"Server {server_id} health check failed")
                        await self._reconnect_server(server_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Health check error: {e}")
    
    async def _check_server_health(self, server_id: str) -> bool:
        if server_id not in self._servers:
            return False
        
        server = self._servers[server_id]
        
        try:
            if server.transport == MCPTransport.STDIO and server.command:
                result = await asyncio.wait_for(
                    self._ping_stdio(server),
                    timeout=5.0
                )
                return result
            return True
        except Exception:
            return False
    
    async def _ping_stdio(self, server: MCPServerConfig) -> bool:
        try:
            process = await asyncio.create_subprocess_exec(
                server.command,
                *(server.args or []),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdin_data = json.dumps({"method": "ping"}).encode()
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(stdin_data),
                    timeout=5.0
                )
                
                return process.returncode == 0
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        except Exception:
            return False
    
    async def _reconnect_server(self, server_id: str) -> bool:
        self._logger.info(f"Attempting to reconnect server: {server_id}")
        
        conn = self._connections.get(server_id)
        if conn:
            conn.state = MCPConnectionState.CONNECTING
        
        try:
            connected = await self.connect(server_id)
            if connected:
                self._circuit_breakers[server_id].record_success()
                return True
        except Exception as e:
            self._logger.error(f"Reconnect failed: {e}")
        
        self._circuit_breakers[server_id].record_failure()
        return False
    
    def list_servers(self) -> List[str]:
        return list(self._servers.keys())
    
    async def list_tools(self, server_id: str) -> List[MCPToolInfo]:
        if server_id in self._tool_cache:
            return self._tool_cache[server_id]
        
        if server_id not in self._servers:
            return []
        
        server = self._servers[server_id]
        
        try:
            if server.transport == MCPTransport.STDIO:
                tools = await self._list_tools_stdio(server)
            elif server.transport == MCPTransport.SSE:
                tools = await self._list_tools_sse(server)
            else:
                tools = []
            
            self._tool_cache[server_id] = tools
            return tools
        except Exception as e:
            self._logger.error(f"Failed to list tools for {server_id}: {e}")
            return []
    
    async def _list_tools_stdio(self, server: MCPServerConfig) -> List[MCPToolInfo]:
        return [
            MCPToolInfo(
                name=f"mcp.{server.server_id}.default",
                description=f"Default tool for {server.server_id}",
                input_schema={"type": "object"}
            )
        ]
    
    async def _list_tools_sse(self, server: MCPServerConfig) -> List[MCPToolInfo]:
        return []
    
    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict,
        timeout_seconds: Optional[int] = None
    ) -> MCPToolResult:
        if server_id not in self._servers:
            return MCPToolResult(
                success=False,
                error=f"Server not found: {server_id}"
            )
        
        breaker = self._circuit_breakers.get(server_id)
        if breaker and breaker.is_open():
            return MCPToolResult(
                success=False,
                error=f"Circuit breaker open for {server_id}"
            )
        
        server = self._servers[server_id]
        timeout = timeout_seconds or self._default_timeout
        
        try:
            if server.transport == MCPTransport.STDIO:
                result = await self._call_stdio(server, tool_name, arguments, timeout)
            elif server.transport == MCPTransport.SSE:
                result = await self._call_sse(server, tool_name, arguments, timeout)
            else:
                result = MCPToolResult(
                    success=False,
                    error=f"Unsupported transport: {server.transport}"
                )
            
            if breaker and result.success:
                breaker.record_success()
            
            return result
            
        except asyncio.TimeoutError:
            if breaker:
                breaker.record_failure()
            return MCPToolResult(
                success=False,
                error=f"Timeout after {timeout} seconds"
            )
        except Exception as e:
            if breaker:
                breaker.record_failure()
            return MCPToolResult(
                success=False,
                error=str(e)
            )
    
    async def _call_stdio(
        self,
        server: MCPServerConfig,
        tool_name: str,
        arguments: dict,
        timeout: int
    ) -> MCPToolResult:
        if not server.command:
            return MCPToolResult(
                success=False,
                error="No command configured for stdio transport"
            )
        
        request = {
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        try:
            process = await asyncio.create_subprocess_exec(
                server.command,
                *(server.args or []),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdin_data = json.dumps(request).encode()
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(stdin_data),
                    timeout=timeout
                )
                
                if process.returncode != 0:
                    return MCPToolResult(
                        success=False,
                        error=f"Process failed with code {process.returncode}: {stderr.decode()}"
                    )
                
                response = json.loads(stdout.decode())
                return MCPToolResult(
                    success=True,
                    output=response.get("result")
                )
                
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                    
        except json.JSONDecodeError as e:
            return MCPToolResult(
                success=False,
                error=f"Invalid JSON response: {e}"
            )
    
    async def _call_sse(
        self,
        server: MCPServerConfig,
        tool_name: str,
        arguments: dict,
        timeout: int
    ) -> MCPToolResult:
        if not server.url:
            return MCPToolResult(
                success=False,
                error="No URL configured for SSE transport"
            )
        
        return MCPToolResult(
            success=False,
            error="SSE transport not implemented yet"
        )
    
    async def health_check(self, server_id: str) -> bool:
        return await self._check_server_health(server_id)
    
    async def connect(self, server_id: str) -> bool:
        if server_id not in self._servers:
            return False
        
        conn = self._connections.get(server_id)
        if conn and conn.state == MCPConnectionState.CONNECTED:
            return True
        
        server = self._servers[server_id]
        
        try:
            if server.transport == MCPTransport.STDIO:
                healthy = await self._check_server_health(server_id)
                if healthy:
                    conn.state = MCPConnectionState.CONNECTED
                    conn.connected_at = time.time()
                    conn.error_count = 0
                    return True
            return False
        except Exception as e:
            self._logger.error(f"Connect failed: {e}")
            if conn:
                conn.state = MCPConnectionState.ERROR
                conn.error_count += 1
            return False
    
    async def disconnect(self, server_id: str) -> None:
        conn = self._connections.get(server_id)
        if conn:
            conn.state = MCPConnectionState.DISCONNECTED
        
        if server_id in self._tool_cache:
            del self._tool_cache[server_id]
    
    def get_server_info(self, server_id: str) -> Optional[MCPServerConfig]:
        return self._servers.get(server_id)
    
    def get_connection_state(self, server_id: str) -> Optional[str]:
        conn = self._connections.get(server_id)
        return conn.state.value if conn else None
    
    async def health(self) -> dict:
        servers_health = {}
        
        for server_id in self._servers:
            healthy = await self._check_server_health(server_id)
            breaker = self._circuit_breakers.get(server_id)
            conn = self._connections.get(server_id)
            
            servers_health[server_id] = {
                "healthy": healthy,
                "circuit_breaker": breaker.state if breaker else "unknown",
                "connection_state": conn.state.value if conn else "unknown",
                "error_count": conn.error_count if conn else 0
            }
        
        return {
            "status": "healthy" if all(s["healthy"] for s in servers_health.values()) else "degraded",
            "servers": servers_health
        }
    
    async def close(self) -> None:
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        for server_id in list(self._connections.keys()):
            await self.disconnect(server_id)
        
        self._logger.info("MCP adapter closed")
