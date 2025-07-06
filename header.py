from contextvars import ContextVar
from typing import Any, Dict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Context variable to store headers across the request
_request_headers: ContextVar[Dict[str, str]] = ContextVar('request_headers', default={})

class HeaderCaptureMiddleware(BaseHTTPMiddleware):
    """Middleware to capture HTTP headers and make them available via context vars."""
    
    async def dispatch(self, request: Request, call_next):
        # Store headers in context variable
        headers = dict(request.headers)
        _request_headers.set(headers)
        
        response = await call_next(request)
        return response

def get_request_headers() -> Dict[str, str]:
    """Get the current request headers from context."""
    return _request_headers.get({})

def get_header(name: str, default: Any = None) -> Any:
    """Get a specific header value."""
    headers = get_request_headers()
    return headers.get(name.lower(), default)

# Modified FastMCP server with header access
class FastMCPWithHeaders(FastMCP):
    """Extended FastMCP with header access capabilities."""
    
    def sse_app(self, mount_path: str | None = None) -> Starlette:
        """Override to add header middleware."""
        app = super().sse_app(mount_path)
        
        # Add our header capture middleware
        app.add_middleware(HeaderCaptureMiddleware)
        
        return app
    
    def streamable_http_app(self) -> Starlette:
        """Override to add header middleware."""
        app = super().streamable_http_app()
        
        # Add our header capture middleware
        app.add_middleware(HeaderCaptureMiddleware)
        
        return app
    
    async def list_tools(self) -> list[MCPTool]:
        """List all available tools with header access."""
        # Access headers
        user_agent = get_header('user-agent')
        authorization = get_header('authorization')
        custom_header = get_header('x-custom-header')
        
        # Log or use headers as needed
        if user_agent:
            print(f"Request from: {user_agent}")
        
        # You can now conditionally modify tools based on headers
        tools = self._tool_manager.list_tools()
        
        # Example: Filter tools based on authorization
        if not authorization:
            # Return limited tools for unauthenticated requests
            tools = [tool for tool in tools if not getattr(tool, 'requires_auth', False)]
        
        return [
            MCPTool(
                name=info.name,
                title=info.title,
                description=info.description,
                inputSchema=info.parameters,
                outputSchema=info.output_schema,
                annotations=info.annotations,
            )
            for info in tools
        ]

# Usage example
if __name__ == "__main__":
    # Create server with header access
    server = FastMCPWithHeaders(name="Header-Aware MCP Server")
    
    @server.tool()
    def example_tool(text: str) -> str:
        """Example tool that can also access headers."""
        user_agent = get_header('user-agent', 'Unknown')
        return f"Processed '{text}' from {user_agent}"
    
    # Run the server
    server.run(transport="streamable-http")