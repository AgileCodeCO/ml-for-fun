using Microsoft.SemanticKernel;

var modelId = "gpt-35-turbo-16k";
var endpoint = "[endpoint]";  
var apiKey = "[apikey]";

var builder = Kernel.CreateBuilder().AddAzureOpenAIChatCompletion(modelId, endpoint, apiKey);

var kernel = builder.Build();

var result = await kernel.InvokePromptAsync("Give me a list of breakfast foods with eggs and cheese");
Console.WriteLine(result); 