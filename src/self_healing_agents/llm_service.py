import os
from typing import List, Dict, Union, Literal, get_args
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
# Assuming langchain_openai and langchain_deepseek are installed
# or langchain_community for ChatDeepseek


load_dotenv()


try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None 
try:
    from langchain_deepseek import ChatDeepSeek
except ImportError:
    ChatDeepSeek = None


SUPPORTED_PROVIDERS_LITERAL = Literal["openai", "deepseek"]
SUPPORTED_PROVIDER_NAMES: List[str] = list(get_args(SUPPORTED_PROVIDERS_LITERAL))

class LLMServiceError(Exception):
    """Custom exception for LLMService errors."""
    pass

class LLMService:
    """
    Abstraction layer for interacting with LLM providers.
    Currently supports OpenAI and Deepseek (via an OpenAI-compatible API or specific SDK).
    """
    def __init__(
        self,
        provider: SUPPORTED_PROVIDERS_LITERAL,
        model_name: str,
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        # base_url: str = None, # Useful for self-hosted or non-standard OpenAI-compatible APIs
    ):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        # self.base_url = base_url # Not used for now, but can be added

        # Check for provider support FIRST
        if self.provider not in get_args(SUPPORTED_PROVIDERS_LITERAL):
            raise LLMServiceError(f"Unsupported LLM provider: {self.provider}")

        if api_key:
            self.api_key = api_key
        else:
            env_var_map = {
                "openai": "OPENAI_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
            }
            env_var = env_var_map.get(self.provider)
            self.api_key = os.getenv(env_var) if env_var else None

        if not self.api_key:
            raise LLMServiceError(
                f"API key for {self.provider} not found. "
                f"Please provide it directly or set the {env_var_map.get(self.provider)} environment variable."
            )

        self._client = self._initialize_client()

    def _initialize_client(self):
        """Initializes the LLM client based on the provider."""
        if self.provider == "openai":
            if ChatOpenAI is None:
                raise LLMServiceError("langchain_openai is not installed. Please install it to use the OpenAI provider.")
            return ChatOpenAI(
                model_name=self.model_name,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # openai_api_base=self.base_url if self.base_url else None, # if using base_url
            )
        elif self.provider == "deepseek":
            if ChatDeepSeek is None:
                raise LLMServiceError("ChatDeepSeek not found. Ensure langchain-deepseek is installed.")
            # LangChain's ChatDeepSeek typically picks up DEEPSEEK_API_KEY from the environment automatically.
            return ChatDeepSeek(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # No explicit api_key or deepseek_api_key here, relies on environment variable
            )
        else:
            raise LLMServiceError(f"Unsupported LLM provider: {self.provider}")

    def _prepare_messages(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        """Converts a list of message dicts to LangChain message objects."""
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                raise LLMServiceError(f"Unsupported message role: {role}")
        return lc_messages

    def invoke(
        self,
        messages: List[Dict[str, str]],
        expect_json: bool = False
    ) -> Union[str, Dict]:
        """
        Sends messages to the LLM and gets a response.

        Args:
            messages: A list of message dictionaries, e.g.,
                      [{"role": "system", "content": "You are helpful."},
                       {"role": "user", "content": "Hello!"}]
            expect_json: If True, attempts to parse the LLM output as JSON.

        Returns:
            The LLM's response as a string, or a dictionary if expect_json is True.
        
        Raises:
            LLMServiceError: If there's an issue with the API call or response parsing.
        """
        if not self._client:
            raise LLMServiceError("LLM client not initialized.")

        lc_messages = self._prepare_messages(messages)

        try:
            if expect_json:
                # For JSON output, we'd ideally use the .with_structured_output() method
                # if the Langchain model supports Pydantic models or JSON schema.
                # For a simpler approach here, we'll just invoke and try to parse.
                # A more robust way is to pass a Pydantic model to with_structured_output.
                # For now, this is a basic implementation.
                response = self._client.invoke(lc_messages)
                content = response.content
                if isinstance(content, str):
                    import json
                    # Attempt to strip markdown fences if present
                    if content.startswith("```json") and content.endswith("```"):
                        content_to_parse = content[len("```json"):-(len("```"))].strip()
                    elif content.startswith("```") and content.endswith("```"):
                        # General case for markdown fence without language specifier
                        content_to_parse = content[len("```"):-(len("```"))].strip()
                    else:
                        content_to_parse = content
                    try:
                        return json.loads(content_to_parse)
                    except json.JSONDecodeError as e:
                        raise LLMServiceError(f"Failed to parse LLM output as JSON: {e}. Raw output: {content_to_parse} (Original: {response.content})")
                elif isinstance(content, dict): # Some models might directly return dicts with structured output
                    return content
                else:
                    raise LLMServiceError(f"Unexpected response content type for JSON: {type(content)}")

            else:
                response = self._client.invoke(lc_messages)
                if hasattr(response, 'content'):
                    return response.content
                else: # Fallback for older or different response structures
                    return str(response)

        except Exception as e:
            # Catching general exceptions from LLM client calls
            raise LLMServiceError(f"Error invoking LLM provider {self.provider}: {e}")

# Example Usage (for demonstration, typically you'd use this in your agents)
if __name__ == '__main__':
    print("LLMService module loaded.")
    # This example won't run without actual API keys set as environment variables
    # e.g., export OPENAI_API_KEY="your_key" or export DEEPSEEK_API_KEY="your_key"

    # Example for OpenAI (requires OPENAI_API_KEY to be set)
    # try:
    #     print("\nAttempting OpenAI Example:")
    #     openai_service = LLMService(provider="openai", model_name="gpt-3.5-turbo")
    #     openai_messages = [
    #         {"role": "system", "content": "You are a helpful assistant that provides concise answers."},
    #         {"role": "user", "content": "What is the capital of France?"}
    #     ]
    #     response_openai = openai_service.invoke(openai_messages)
    #     print(f"OpenAI Response: {response_openai}")

    #     openai_messages_json = [
    #         {"role": "system", "content": "You are a helpful assistant. Respond in JSON format."},
    #         {"role": "user", "content": "Provide details for a user with name 'John Doe' and age 30."}
    #     ]
    #     # To make this work reliably, the prompt must strongly guide the LLM to output JSON.
    #     # E.g., "Respond with a JSON object containing keys 'name' and 'age'."
    #     response_openai_json = openai_service.invoke(openai_messages_json, expect_json=True)
    #     print(f"OpenAI JSON Response: {response_openai_json}")

    # except LLMServiceError as e:
    #     print(f"OpenAI Example Error: {e}")
    # except Exception as e:
    #     print(f"An unexpected error occurred in OpenAI example: {e}")


    # Example for Deepseek (requires DEEPSEEK_API_KEY to be set)
    #try:
    #     print("\nAttempting Deepseek Example:")
    #    # Ensure you have the correct model name for Deepseek
    #    deepseek_service = LLMService(provider="deepseek", model_name="deepseek-chat") # Or other valid model
    #    deepseek_messages = [
    #        {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    #        {"role": "user", "content": "Explain recursion in Python."}
    #    ]
    #    response_deepseek = deepseek_service.invoke(deepseek_messages)
    #    print(f"Deepseek Response: {response_deepseek}")
    #except LLMServiceError as e:
    #    print(f"Deepseek Example Error: {e}")
    #except Exception as e:
    #    print(f"An unexpected error occurred in Deepseek example: {e}")
    
    print("\nTo run examples, uncomment them and ensure API keys (OPENAI_API_KEY, DEEPSEEK_API_KEY) are set in your environment.") 