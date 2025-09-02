import asyncio

from semantic_kernel import Kernel
from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

import os
from dotenv import load_dotenv

import logging
from lights_plugin import LightsPlugin

async def main():
    # Load environment variables from .env file
    load_dotenv()
    # Get environment variables
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    model_deployment = os.getenv("MODEL_DEPLOYMENT")
    api_key = os.getenv("API_KEY")
    
    print(f"Project Endpoint: {project_endpoint}")
    print(f"Model Deployment: {model_deployment}")
    print(f"API Key: {api_key}")

    # Init the kernel
    kernel = Kernel()
    
    chat_completion = AzureChatCompletion(
        deployment_name=model_deployment,
        api_key=api_key,
        base_url=project_endpoint
    )
    
    kernel.add_service(chat_completion)

    # Set logging level
    setup_logging()
    logging.getLogger("kernel").setLevel(logging.DEBUG)
    
    kernel.add_plugin(
        LightsPlugin(),
        plugin_name="Lights"
    )
    
    # Enable planning
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    
    # History
    history = ChatHistory()
    
    userInput = None
    while True:
        userInput = input("User > ")
        
        if userInput.lower() == "exit":
            break
        
        history.add_user_message(userInput)
        
        result = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel = kernel
        )
        
        print ("Assistant > " + str(result))
        
        history.add_message(result)
        
if __name__ == "__main__":
    asyncio.run(main())
        


