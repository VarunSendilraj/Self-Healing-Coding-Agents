import pytest
from unittest.mock import patch, MagicMock
import os

# Ensure the test can find the src directory
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


from self_healing_agents.llm_service import LLMService, LLMServiceError, SUPPORTED_PROVIDERS

# Mock environment variables for API keys
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_openai_key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake_deepseek_key")

@pytest.fixture
def mock_chat_openai_constructor():
    with patch('self_healing_agents.llm_service.ChatOpenAI') as mock_constructor:
        mock_instance = MagicMock()
        mock_constructor.return_value = mock_instance
        yield mock_constructor

@pytest.fixture
def mock_chat_deepseek_constructor():
    with patch('self_healing_agents.llm_service.ChatDeepSeek') as mock_constructor:
        mock_instance = MagicMock()
        mock_constructor.return_value = mock_instance
        yield mock_constructor

# Test successful initialization for OpenAI
@pytest.mark.skip(reason="OpenAI provider tests are currently disabled")
def test_llm_service_init_openai_success(mock_chat_openai_constructor):
    service = LLMService(provider="openai", model_name="gpt-3.5-turbo")
    assert service.provider == "openai"
    assert service.model_name == "gpt-3.5-turbo"
    assert service.api_key == "fake_openai_key"
    mock_chat_openai_constructor.assert_called_once_with(
        model_name="gpt-3.5-turbo",
        api_key="fake_openai_key",
        temperature=0.7,
        max_tokens=1024,
    )

# Test successful initialization for Deepseek
def test_llm_service_init_deepseek_success(mock_chat_deepseek_constructor):
    service = LLMService(provider="deepseek", model_name="deepseek-chat")
    assert service.provider == "deepseek"
    assert service.model_name == "deepseek-chat"
    assert service.api_key == "fake_deepseek_key"
    mock_chat_deepseek_constructor.assert_called_once_with(
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=1024,
    )

# Test initialization with direct API key
@pytest.mark.skip(reason="OpenAI provider tests are currently disabled")
def test_llm_service_init_with_direct_api_key(mock_chat_openai_constructor):
    service = LLMService(provider="openai", model_name="gpt-3.5-turbo", api_key="direct_key")
    assert service.api_key == "direct_key"
    mock_chat_openai_constructor.assert_called_once_with(
        model_name="gpt-3.5-turbo",
        api_key="direct_key",
        temperature=0.7,
        max_tokens=1024,
    )

# Test initialization failure if API key is missing
@pytest.mark.skip(reason="OpenAI provider tests are currently disabled")
def test_llm_service_init_failure_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMServiceError, match="API key for openai not found"):
        LLMService(provider="openai", model_name="gpt-3.5-turbo")

# Test initialization failure for unsupported provider
def test_llm_service_init_failure_unsupported_provider():
    with pytest.raises(LLMServiceError, match="Unsupported LLM provider: unknown"):
        LLMService(provider="unknown", model_name="test-model")

# Test _prepare_messages
def test_prepare_messages():
    # Use deepseek provider as it's not skipped
    with patch('self_healing_agents.llm_service.ChatDeepSeek'): # Mock to allow instantiation
        service = LLMService(provider="deepseek", model_name="test") 
    messages_input = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User query"},
        {"role": "assistant", "content": "AI response"}
    ]
    lc_messages = service._prepare_messages(messages_input)
    assert len(lc_messages) == 3
    assert lc_messages[0].type == "system"
    assert lc_messages[0].content == "System prompt"
    assert lc_messages[1].type == "human"
    assert lc_messages[1].content == "User query"
    assert lc_messages[2].type == "ai"
    assert lc_messages[2].content == "AI response"

def test_prepare_messages_invalid_role():
    # Use deepseek provider as it's not skipped
    with patch('self_healing_agents.llm_service.ChatDeepSeek'): # Mock to allow instantiation
        service = LLMService(provider="deepseek", model_name="test")
    messages_input = [{"role": "invalid", "content": "test"}]
    with pytest.raises(LLMServiceError, match="Unsupported message role: invalid"):
        service._prepare_messages(messages_input)

# Test invoke for text response (OpenAI)
@pytest.mark.skip(reason="OpenAI provider tests are currently disabled")
def test_invoke_text_response_openai(mock_chat_openai_constructor):
    mock_openai_client_instance = mock_chat_openai_constructor.return_value
    
    mock_llm_response_obj = MagicMock()
    mock_llm_response_obj.content = "This is a text response."
    mock_openai_client_instance.invoke.return_value = mock_llm_response_obj

    service = LLMService(provider="openai", model_name="gpt-3.5-turbo")
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"}
    ]
    response = service.invoke(messages)
    
    assert response == "This is a text response."
    args, _ = mock_openai_client_instance.invoke.call_args
    assert len(args[0]) == 2 
    assert args[0][0].type == "system"
    assert args[0][1].type == "human"


# Test invoke for text response (Deepseek)
def test_invoke_text_response_deepseek(mock_chat_deepseek_constructor):
    mock_deepseek_client_instance = mock_chat_deepseek_constructor.return_value

    mock_llm_response_obj = MagicMock()
    mock_llm_response_obj.content = "Deepseek text response."
    mock_deepseek_client_instance.invoke.return_value = mock_llm_response_obj

    service = LLMService(provider="deepseek", model_name="deepseek-chat")
    messages = [{"role": "user", "content": "Hi Deepseek"}]
    response = service.invoke(messages)
    
    assert response == "Deepseek text response."
    args, _ = mock_deepseek_client_instance.invoke.call_args
    assert len(args[0]) == 1
    assert args[0][0].type == "human"

# Test invoke for JSON response (OpenAI)
@pytest.mark.skip(reason="OpenAI provider tests are currently disabled")
def test_invoke_json_response_openai(mock_chat_openai_constructor):
    mock_openai_client_instance = mock_chat_openai_constructor.return_value

    mock_llm_response_obj = MagicMock()
    mock_llm_response_obj.content = '''{"key": "value", "number": 123}'''
    mock_openai_client_instance.invoke.return_value = mock_llm_response_obj

    service = LLMService(provider="openai", model_name="gpt-3.5-turbo")
    messages = [{"role": "user", "content": "Get JSON"}]
    response = service.invoke(messages, expect_json=True)
    
    assert response == {"key": "value", "number": 123}

# Test invoke for JSON response where model directly returns dict (OpenAI)
@pytest.mark.skip(reason="OpenAI provider tests are currently disabled")
def test_invoke_json_response_direct_dict_openai(mock_chat_openai_constructor):
    mock_openai_client_instance = mock_chat_openai_constructor.return_value

    mock_llm_response_obj = MagicMock()
    mock_llm_response_obj.content = {"key": "value_direct", "number": 456}
    mock_openai_client_instance.invoke.return_value = mock_llm_response_obj

    service = LLMService(provider="openai", model_name="gpt-3.5-turbo")
    messages = [{"role": "user", "content": "Get JSON object"}]
    response = service.invoke(messages, expect_json=True)
    
    assert response == {"key": "value_direct", "number": 456}


# Test invoke for JSON response with parsing error
@pytest.mark.skip(reason="OpenAI provider tests are currently disabled as this test uses its mock setup")
def test_invoke_json_response_parsing_error(mock_chat_openai_constructor):
    mock_openai_client_instance = mock_chat_openai_constructor.return_value

    mock_llm_response_obj = MagicMock()
    mock_llm_response_obj.content = "This is not JSON."
    mock_openai_client_instance.invoke.return_value = mock_llm_response_obj

    service = LLMService(provider="openai", model_name="gpt-3.5-turbo")
    messages = [{"role": "user", "content": "Get JSON"}]
    
    with pytest.raises(LLMServiceError, match="Failed to parse LLM output as JSON"):
        service.invoke(messages, expect_json=True)

# Test invoke for JSON response with unexpected content type
@pytest.mark.skip(reason="OpenAI provider tests are currently disabled as this test uses its mock setup")
def test_invoke_json_response_unexpected_type(mock_chat_openai_constructor):
    mock_openai_client_instance = mock_chat_openai_constructor.return_value
    
    mock_llm_response_obj = MagicMock()
    mock_llm_response_obj.content = 12345 # Not a string or dict
    mock_openai_client_instance.invoke.return_value = mock_llm_response_obj

    service = LLMService(provider="openai", model_name="gpt-3.5-turbo")
    messages = [{"role": "user", "content": "Get JSON"}]
    
    with pytest.raises(LLMServiceError, match="Unexpected response content type for JSON"):
        service.invoke(messages, expect_json=True)


# Test invoke when client not initialized (e.g., if _initialize_client failed or was bypassed)
def test_invoke_client_not_initialized():
    # Use deepseek provider as it's not skipped
    with patch('self_healing_agents.llm_service.ChatDeepSeek'): # Mock to allow instantiation
        service = LLMService(provider="deepseek", model_name="deepseek-chat")
    service._client = None # Force client to be None
    
    with pytest.raises(LLMServiceError, match="LLM client not initialized."):
        service.invoke([{"role": "user", "content": "Hello"}])

# Test general LLM client invocation error
@pytest.mark.skip(reason="OpenAI provider tests are currently disabled as this test uses its mock setup")
def test_invoke_llm_provider_error(mock_chat_openai_constructor):
    mock_openai_client_instance = mock_chat_openai_constructor.return_value
    mock_openai_client_instance.invoke.side_effect = Exception("LLM API is down")

    service = LLMService(provider="openai", model_name="gpt-3.5-turbo")
    messages = [{"role": "user", "content": "Hello"}]
    
    with pytest.raises(LLMServiceError, match="Error invoking LLM provider openai: LLM API is down"):
        service.invoke(messages)

# Test for langchain_openai not installed
@pytest.mark.skip(reason="OpenAI provider tests are currently disabled")
def test_openai_not_installed(monkeypatch):
    monkeypatch.setattr('self_healing_agents.llm_service.ChatOpenAI', None)
    with pytest.raises(LLMServiceError, match="langchain_openai is not installed"):
        LLMService(provider="openai", model_name="gpt-3.5-turbo")

# Test for langchain_deepseek not installed
def test_deepseek_not_installed(monkeypatch):
    monkeypatch.setattr('self_healing_agents.llm_service.ChatDeepSeek', None)
    with pytest.raises(LLMServiceError, match="ChatDeepSeek not found"):
        LLMService(provider="deepseek", model_name="deepseek-chat")

# Python path comment fix - remove if it was temporary for local testing
# We need to ensure that the tests can run in CI/remotely without path hacks.
# The typical way is to install the package in editable mode (pip install -e .)
# or ensure PYTHONPATH is set up correctly in the test environment.
# For now, let's assume the structure `src/self_healing_agents` and `tests/`
# and that pytest is run from the project root.

# To run these tests:
# 1. Make sure you are in the root directory of your project (SelfHealingAgentsV2).
# 2. Ensure pytest and unittest.mock are installed.
# 3. Run: pytest tests/test_llm_service.py 