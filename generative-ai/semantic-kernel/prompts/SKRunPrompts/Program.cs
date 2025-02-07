using Microsoft.SemanticKernel;

var modelId = "gpt-35-turbo-16k";
var endpoint = "https://rag-research.openai.azure.com/";  
var apiKey = "080ddc88dc5344b78604b8d5d5f0389b";

var builder = Kernel.CreateBuilder();
builder.AddAzureOpenAIChatCompletion(modelId, endpoint, apiKey);

var kernel = builder.Build();

// Prompt and variables
var prompt = """
   You are a helpful travel guide.
   I'm visiting {{$city}}. {{$background}}. What are some activities I should do today?
   """;

var city = "Barcelona";
var background = "I really enjoy art and dance";

// Kernel function
var activitiesFunction = kernel.CreateFunctionFromPrompt(prompt);
var arguments = new KernelArguments
{
    ["city"] = city ,
    ["background"] = background
};

// Invoke the function
var result = await kernel.InvokeAsync(activitiesFunction, arguments);
Console.WriteLine(result);
