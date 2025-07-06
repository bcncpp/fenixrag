#!/usr/bin/env python3
"""
MCP Client with support for different transports and custom headers.
Supports stdio, SSE, and StreamableHTTP transports.
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import SseClientTransport
from mcp.types import (
    CallToolRequest,
    GetPromptRequest,
    ListPromptsRequest,
    ListResourcesRequest,
    ListToolsRequest,
    ReadResourceRequest,
)


class MCPClient:
    """A comprehensive MCP client supporting multiple transports."""
    
    def __init__(self, base_url: str = "http://localhost:8000", headers: Optional[Dict[str, str]] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.session: Optional[ClientSession] = None
        self.transport = None
        
    async def connect_stdio(self, command: str, args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None):
        """Connect using stdio transport (subprocess)."""
        params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env
        )
        
        stdio_transport = stdio_client(params)
        self.session, self.transport = await stdio_transport.__aenter__()
        
        # Initialize the session
        await self.session.initialize()
        
    async def connect_sse(self, sse_endpoint: str = "/sse"):
        """Connect using SSE transport."""
        sse_url = urljoin(self.base_url, sse_endpoint)
        
        transport = SseClientTransport(sse_url, headers=self.headers)
        self.session, self.transport = await transport.__aenter__()
        
        # Initialize the session
        await self.session.initialize()
        
    async def connect_streamable_http(self, endpoint: str = "/mcp"):
        """Connect using StreamableHTTP transport."""
        from mcp.client.streamable_http import StreamableHTTPClientTransport
        
        http_url = urljoin(self.base_url, endpoint)
        
        transport = StreamableHTTPClientTransport(http_url, headers=self.headers)
        self.session, self.transport = await transport.__aenter__()
        
        # Initialize the session
        await self.session.initialize()
        
    async def disconnect(self):
        """Disconnect from the server."""
        if self.transport:
            await self.transport.__aexit__(None, None, None)
            
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        if not self.session:
            raise RuntimeError("Not connected to server")
            
        result = await self.session.list_tools(ListToolsRequest())
        return [tool.model_dump() for tool in result.tools]
        
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call a tool with given arguments."""
        if not self.session:
            raise RuntimeError("Not connected to server")
            
        result = await self.session.call_tool(
            CallToolRequest(name=name, arguments=arguments)
        )
        return [content.model_dump() for content in result.content]
        
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources."""
        if not self.session:
            raise RuntimeError("Not connected to server")
            
        result = await self.session.list_resources(ListResourcesRequest())
        return [resource.model_dump() for resource in result.resources]
        
    async def read_resource(self, uri: str) -> List[Dict[str, Any]]:
        """Read a resource by URI."""
        if not self.session:
            raise RuntimeError("Not connected to server")
            
        result = await self.session.read_resource(ReadResourceRequest(uri=uri))
        return [content.model_dump() for content in result.contents]
        
    async def list_prompts(self) -> List[Dict[str, Any]]:
        """List available prompts."""
        if not self.session:
            raise RuntimeError("Not connected to server")
            
        result = await self.session.list_prompts(ListPromptsRequest())
        return [prompt.model_dump() for prompt in result.prompts]
        
    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get a prompt with given arguments."""
        if not self.session:
            raise RuntimeError("Not connected to server")
            
        result = await self.session.get_prompt(
            GetPromptRequest(name=name, arguments=arguments or {})
        )
        return result.model_dump()


class SimpleHTTPMCPClient:
    """A simple HTTP-based MCP client using direct HTTP requests."""
    
    def __init__(self, base_url: str = "http://localhost:8000", headers: Optional[Dict[str, str]] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.client = httpx.AsyncClient(headers=self.headers)
        
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
        
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List tools via direct HTTP request."""
        response = await self.client.post(
            f"{self.base_url}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            }
        )
        response.raise_for_status()
        data = response.json()
        return data.get("result", {}).get("tools", [])
        
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call a tool via direct HTTP request."""
        response = await self.client.post(
            f"{self.base_url}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": name,
                    "arguments": arguments
                }
            }
        )
        response.raise_for_status()
        data = response.json()
        return data.get("result", {}).get("content", [])


async def demo_client():
    """Demonstrate the MCP client functionality."""
    
    # Custom headers to send with requests
    custom_headers = {
        "User-Agent": "MCP-Demo-Client/1.0",
        "X-Custom-Header": "demo-value",
        "X-OpenAPI": "3.0.0",
        "Authorization": "Bearer demo-token"
    }
    
    print("🚀 MCP Client Demo")
    print("==================")
    
    # Try StreamableHTTP client first
    try:
        print("\n📡 Connecting via StreamableHTTP...")
        client = MCPClient(headers=custom_headers)
        await client.connect_streamable_http()
        
        print("✅ Connected successfully!")
        
        # List tools
        print("\n🔧 Available Tools:")
        tools = await client.list_tools()
        for tool in tools:
            print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
            
        # List resources
        print("\n📁 Available Resources:")
        try:
            resources = await client.list_resources()
            for resource in resources:
                print(f"  - {resource['uri']}: {resource.get('description', 'No description')}")
        except Exception as e:
            print(f"  ❌ Error listing resources: {e}")
            
        # List prompts
        print("\n💬 Available Prompts:")
        try:
            prompts = await client.list_prompts()
            for prompt in prompts:
                print(f"  - {prompt['name']}: {prompt.get('description', 'No description')}")
        except Exception as e:
            print(f"  ❌ Error listing prompts: {e}")
            
        # Try calling a tool if any exist
        if tools:
            first_tool = tools[0]
            print(f"\n🔨 Testing tool: {first_tool['name']}")
            try:
                # Create sample arguments based on the tool's schema
                sample_args = {}
                if 'inputSchema' in first_tool and 'properties' in first_tool['inputSchema']:
                    for prop, details in first_tool['inputSchema']['properties'].items():
                        if details.get('type') == 'string':
                            sample_args[prop] = "test"
                        elif details.get('type') == 'integer':
                            sample_args[prop] = 42
                        elif details.get('type') == 'number':
                            sample_args[prop] = 3.14
                        elif details.get('type') == 'boolean':
                            sample_args[prop] = True
                            
                result = await client.call_tool(first_tool['name'], sample_args)
                print(f"  ✅ Result: {result}")
            except Exception as e:
                print(f"  ❌ Error calling tool: {e}")
                
        await client.disconnect()
        
    except Exception as e:
        print(f"❌ StreamableHTTP connection failed: {e}")
        
        # Try simple HTTP client as fallback
        print("\n📡 Trying simple HTTP client...")
        try:
            simple_client = SimpleHTTPMCPClient(headers=custom_headers)
            tools = await simple_client.list_tools()
            print(f"✅ Found {len(tools)} tools via simple HTTP")
            for tool in tools:
                print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
            await simple_client.close()
        except Exception as e2:
            print(f"❌ Simple HTTP also failed: {e2}")
            
    # Try SSE client
    print("\n📡 Trying SSE connection...")
    try:
        sse_client = MCPClient(headers=custom_headers)
        await sse_client.connect_sse()
        tools = await sse_client.list_tools()
        print(f"✅ SSE: Found {len(tools)} tools")
        await sse_client.disconnect()
    except Exception as e:
        print(f"❌ SSE connection failed: {e}")


async def interactive_client():
    """Interactive MCP client for testing."""
    custom_headers = {
        "User-Agent": "Interactive-MCP-Client/1.0",
        "X-OpenAPI": "3.0.0",
        "Authorization": "Bearer interactive-token"
    }
    
    print("🔧 Interactive MCP Client")
    print("========================")
    print("Commands:")
    print("  tools    - List available tools")
    print("  call     - Call a tool")
    print("  resources - List resources")
    print("  prompts  - List prompts")
    print("  quit     - Exit")
    print()
    
    client = MCPClient(headers=custom_headers)
    
    try:
        await client.connect_streamable_http()
        print("✅ Connected to MCP server\n")
        
        while True:
            try:
                command = input("mcp> ").strip().lower()
                
                if command == "quit":
                    break
                elif command == "tools":
                    tools = await client.list_tools()
                    print(f"\n📋 Available Tools ({len(tools)}):")
                    for i, tool in enumerate(tools, 1):
                        print(f"{i}. {tool['name']}")
                        print(f"   Description: {tool.get('description', 'No description')}")
                        if 'inputSchema' in tool:
                            props = tool['inputSchema'].get('properties', {})
                            if props:
                                print(f"   Parameters: {', '.join(props.keys())}")
                        print()
                        
                elif command == "call":
                    tools = await client.list_tools()
                    if not tools:
                        print("❌ No tools available")
                        continue
                        
                    print("Available tools:")
                    for i, tool in enumerate(tools, 1):
                        print(f"{i}. {tool['name']}")
                        
                    try:
                        choice = int(input("Select tool number: ")) - 1
                        if choice < 0 or choice >= len(tools):
                            print("❌ Invalid tool number")
                            continue
                            
                        tool = tools[choice]
                        print(f"\nCalling tool: {tool['name']}")
                        
                        # Get arguments
                        args = {}
                        if 'inputSchema' in tool and 'properties' in tool['inputSchema']:
                            for prop, details in tool['inputSchema']['properties'].items():
                                value = input(f"Enter {prop} ({details.get('type', 'any')}): ")
                                if details.get('type') == 'integer':
                                    args[prop] = int(value)
                                elif details.get('type') == 'number':
                                    args[prop] = float(value)
                                elif details.get('type') == 'boolean':
                                    args[prop] = value.lower() in ('true', 'yes', '1')
                                else:
                                    args[prop] = value
                                    
                        result = await client.call_tool(tool['name'], args)
                        print(f"\n✅ Result:")
                        print(json.dumps(result, indent=2))
                        
                    except (ValueError, IndexError):
                        print("❌ Invalid input")
                    except Exception as e:
                        print(f"❌ Error calling tool: {e}")
                        
                elif command == "resources":
                    resources = await client.list_resources()
                    print(f"\n📁 Available Resources ({len(resources)}):")
                    for resource in resources:
                        print(f"  - {resource['uri']}")
                        print(f"    Name: {resource.get('name', 'Unnamed')}")
                        print(f"    Description: {resource.get('description', 'No description')}")
                        print()
                        
                elif command == "prompts":
                    prompts = await client.list_prompts()
                    print(f"\n💬 Available Prompts ({len(prompts)}):")
                    for prompt in prompts:
                        print(f"  - {prompt['name']}")
                        print(f"    Description: {prompt.get('description', 'No description')}")
                        print()
                        
                else:
                    print("❌ Unknown command. Use: tools, call, resources, prompts, or quit")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
    finally:
        await client.disconnect()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_client())
    else:
        asyncio.run(demo_client()):
