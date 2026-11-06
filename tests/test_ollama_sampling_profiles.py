"""Per-model sampling and JSON guidance must reach the native request."""
import pytest
from pydantic import BaseModel

from core_lib.llm.provider_registry import ProviderConfig
from core_lib.llm.providers.ollama_provider import OllamaConfig, OllamaProvider


class Answer(BaseModel):
    answer: str


@pytest.mark.parametrize('mode', ['template', 'native'])
def test_registry_settings_reach_native_ollama_payload(monkeypatch, mode):
    configured = ProviderConfig.from_dict({
        'provider':'ollama','model':'bonsai-2:9b','temperature':.7,
        'top_p':.8,'top_k':20,'min_p':0.0,'presence_penalty':1.5,
        'repeat_penalty':1.0,'repeat_last_n':64,'seed':0,'json_prompt_mode':mode,
    })
    payloads = []
    def send(self, payload, timeout=None):
        payloads.append(payload)
        return {'message':{'content':'{"answer":"ok"}'},'eval_count':5,'prompt_eval_count':10}
    monkeypatch.setattr(OllamaProvider, '_chat_once', send)
    client = configured.to_client()
    source = [{'role':'user','content':'Return the answer.'}]
    response = client.chat(messages=source, structured_output=Answer, thinking_enabled=False)
    assert response['content'] == {'answer':'ok'}
    payload = payloads[0]
    assert payload['think'] is False and isinstance(payload['format'], dict)
    assert payload['options'] == {
        'temperature':.7,'top_p':.8,'top_k':20,'min_p':0.0,'presence_penalty':1.5,
        'repeat_penalty':1.0,'repeat_last_n':64,'seed':0,
    }
    assert source[0]['content'] == 'Return the answer.'
    text = payload['messages'][-1]['content']
    assert (text == source[0]['content']) == (mode == 'native')


def test_unconfigured_profiles_keep_existing_options():
    provider = OllamaProvider.__new__(OllamaProvider)
    provider.config = OllamaConfig(model='ministral-3', temperature=.1)
    assert provider.config.json_prompt_mode == 'template'
    assert provider._build_options(False) == {'temperature':.1}


def test_environment_profile_keeps_zero_values(monkeypatch):
    for name, value in {
        'OLLAMA_MIN_P':'0','OLLAMA_PRESENCE_PENALTY':'0',
        'OLLAMA_REPEAT_LAST_N':'64','OLLAMA_SEED':'0','OLLAMA_JSON_PROMPT_MODE':'native',
    }.items():
        monkeypatch.setenv(name,value)
    config=OllamaConfig.from_env()
    assert config.min_p == 0 and config.presence_penalty == 0
    assert config.seed == 0 and config.repeat_last_n == 64
    assert config.json_prompt_mode == 'native'


def test_native_mode_does_not_silently_drop_schema_constraint(monkeypatch):
    payloads=[]
    def fail(self,payload,timeout=None):
        payloads.append(payload)
        raise RuntimeError('failed to initialize samplers: failed to parse grammar')
    monkeypatch.setattr(OllamaProvider,'_chat_once',fail)
    provider=OllamaProvider(OllamaConfig(model='bonsai-2:9b',json_prompt_mode='native'))
    result=provider.chat(messages=[{'role':'user','content':'Return an answer.'}],structured_output=Answer)
    assert len(payloads)==1 and result['error'] and result['content'] is None


def test_invalid_json_prompt_mode_fails_explicitly():
    with pytest.raises(ValueError,match='json_prompt_mode'):
        OllamaConfig(json_prompt_mode='typo')
