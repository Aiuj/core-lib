from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from core_lib.llm.llm_client import LLMClient
from core_lib.llm.providers.openai_provider import OpenAIConfig


class Answer(BaseModel):
    answer: str


def test_direct_client_required_structured_contract():
    provider = MagicMock()
    provider.chat.return_value = {'content': 'private invalid text', 'structured': False, 'usage': {'completion_tokens': 3}}
    with patch.object(LLMClient, '_initialize_provider', return_value=provider), patch('core_lib.llm.llm_client.setup_tracing'):
        client = LLMClient(OpenAIConfig(api_key='test', model='test'))
        assert client.chat('x', structured_output=Answer)['content'] == 'private invalid text'
        result = client.chat('x', structured_output=Answer, require_structured=True)
        assert result['error_code'] == 'structured_output_invalid'
        assert result['content'] is None and result['usage']['completion_tokens'] == 3
        provider.chat.return_value = {'content': {'answer': 'ok'}, 'structured': True}
        assert client.chat('x', structured_output=Answer, require_structured=True)['content'] == {'answer': 'ok'}
        with pytest.raises(ValueError, match='requires structured_output'):
            client.chat('x', require_structured=True)
