import os
import asyncio
from dotenv import load_dotenv
from agent_framework.azure import AzureAIClient
from agent_framework import MCPStdioTool
from azure.identity.aio import AzureCliCredential

# Load environment variables from .env file
load_dotenv()
# Get environment variables
project_endpoint = os.getenv("PROJECT_ENDPOINT")
model_deployment = os.getenv("MODEL_DEPLOYMENT")

async def main():
    async with (
        MCPStdioTool(
            name="echo-tool", 
            command="dotnet", 
            args=["run", "--project", "/Users/agilecode/Repos/ml-for-fun/mcp/server/AgileCode.AI.ForFun.MCP.Server/AgileCode.AI.ForFun.MCP.ConsoleServer/AgileCode.AI.ForFun.MCP.ConsoleServer.csproj"]
        ) as mcp_server,
        AzureCliCredential() as credential,
        AzureAIClient(credential=credential, project_endpoint=project_endpoint, model_deployment_name=model_deployment).create_agent(
            instructions="You are a testing agent", 
            name="testing-agent"
        ) as agent,
    ):
        result = await agent.run("Test echo tool with input: MCP for win!", tools=[mcp_server])
        print(result.text)

if __name__ == "__main__":
    asyncio.run(main())
