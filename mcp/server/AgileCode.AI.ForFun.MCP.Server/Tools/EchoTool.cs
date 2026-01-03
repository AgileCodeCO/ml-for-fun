using System;
using System.ComponentModel;
using ModelContextProtocol.Server;

namespace AgileCode.AI.ForFun.MCP.Server.Tools;

[McpServerToolType]
public static class EchoTool
{
    [McpServerTool, Description("Echoes the input string back to the caller.")]
    public static string Echo(string input)
    {
        return $"Hello {input}";
    }

}
