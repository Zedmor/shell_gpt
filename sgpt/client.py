import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional

import boto3
import requests
from langchain_community.llms.bedrock import Bedrock, LLMInputOutputAdapter
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk

from .cache import Cache
from .config import cfg

CACHE_LENGTH = int(cfg.get("CACHE_LENGTH"))
CACHE_PATH = Path(cfg.get("CACHE_PATH"))
REQUEST_TIMEOUT = int(cfg.get("REQUEST_TIMEOUT"))


class OpenAIClient:
    cache = Cache(CACHE_LENGTH, CACHE_PATH)

    def __init__(self, api_host: str, api_key: str) -> None:
        self.__api_key = api_key
        self.api_host = api_host

    @cache
    def _request(
            self,
            messages: List[Dict[str, str]],
            model: str = "gpt-4-turbo-2024-0409",
            temperature: float = 1,
            top_probability: float = 1,
            ) -> Generator[str, None, None]:
        """
        Make request to OpenAI API, read more:
        https://platform.openai.com/docs/api-reference/chat

        :param messages: List of messages {"role": user or assistant, "content": message_string}
        :param model: String gpt-4 or gpt-4-0301
        :param temperature: Float in 0.0 - 1.0 range.
        :param top_probability: Float in 0.0 - 1.0 range.
        :return: Response body JSON.
        """
        data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "top_p": top_probability,
            "stream": True,
            }
        endpoint = f"{self.api_host}/v1/chat/completions"
        response = requests.post(
            endpoint,
            # Hide API key from Rich traceback.
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.__api_key}",
                },
            json=data,
            timeout=REQUEST_TIMEOUT,
            stream=True,
            )
        response.raise_for_status()
        # TODO: Optimise.
        # https://github.com/openai/openai-python/blob/237448dc072a2c062698da3f9f512fae38300c1c
        # /openai/api_requestor.py#L98
        for line in response.iter_lines():
            data = line.lstrip(b"data: ").decode("utf-8")
            if data == "[DONE]":  # type: ignore
                break
            if not data:
                continue
            data = json.loads(data)  # type: ignore
            delta = data["choices"][0]["delta"]  # type: ignore
            if "content" not in delta:
                continue
            yield delta["content"]

    def get_completion(
            self,
            messages: List[Dict[str, str]],
            model: str = "gpt-4",
            temperature: float = 1,
            top_probability: float = 1,
            caching: bool = True,
            ) -> Generator[str, None, None]:
        """
        Generates single completion for prompt (message).

        :param messages: List of dict with messages and roles.
        :param model: String gpt-4 or gpt-4-0301.
        :param temperature: Float in 0.0 - 1.0 range.
        :param top_probability: Float in 0.0 - 1.0 range.
        :param caching: Boolean value to enable/disable caching.
        :return: String generated completion.
        """
        yield from self._request(
            messages,
            model,
            temperature,
            top_probability,
            caching=caching,
            )


class ModifiedBedrock(Bedrock):
    def _stream_messages(self,
                         messages: List,
                         stop: Optional[List[str]] = None,
                         run_manager: Optional[CallbackManagerForLLMRun] = None,
                         **kwargs: Any):
        return self._prepare_input_and_invoke_stream_messages(
            messages=messages, stop=stop, run_manager=run_manager, **kwargs
            )

    @classmethod
    def prepare_output_stream(
            cls, provider: str, response: Any, stop: Optional[List[str]] = None
            ) -> Iterator[GenerationChunk]:
        stream = response.get("body")

        if not stream:
            return

        if provider not in LLMInputOutputAdapter.provider_to_output_key_map:
            raise ValueError(
                f"Unknown streaming response output key for provider: {provider}"
                )

        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                chunk_obj = json.loads(chunk.get("bytes").decode())
                if provider == "cohere" and (
                        chunk_obj["is_finished"]
                        or chunk_obj[LLMInputOutputAdapter.provider_to_output_key_map[provider]]
                        == "<EOS_TOKEN>"
                ):
                    return

                # chunk obj format varies with provider
                if chunk_obj['type'] == 'content_block_delta':
                    yield GenerationChunk(text=chunk_obj['delta']['text'])

    def _prepare_input_and_invoke_stream_messages(
            self,
            messages: List,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
            ) -> Iterator[GenerationChunk]:
        _model_kwargs = self.model_kwargs or {}
        provider = self._get_provider()

        if stop:
            if provider not in self.provider_stop_sequence_key_name_map:
                raise ValueError(
                    f"Stop sequence key name for {provider} is not supported."
                    )

            # stop sequence from _generate() overrides
            # stop sequences in the class attribute
            _model_kwargs[self.provider_stop_sequence_key_name_map.get(provider)] = stop

        if provider == "cohere":
            _model_kwargs["stream"] = True

        params = {**_model_kwargs, **kwargs}
        input_body = {"max_tokens": params['max_tokens_to_sample']}
        input_body['messages'] = [
            {'role': m['role'], 'content': [{'type': 'text', 'text': m['content']}]} for m in
            messages]
        input_body['anthropic_version'] = 'bedrock-2023-05-31'
        body = json.dumps(input_body)

        try:
            response = self.client.invoke_model_with_response_stream(
                body=body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
                )
        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")

        for chunk in self.prepare_output_stream(
                provider, response, stop
                ):
            yield chunk
            if run_manager is not None:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)


class BedrockClient(OpenAIClient):

    def __init__(self, *args, **kwargs):
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            )

        self.llm = ModifiedBedrock(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            client=self.bedrock_runtime,
            credentials_profile_name="default",
            )

    def _request(
            self,
            messages: List[Dict[str, str]],
            model: str = "gpt-4",
            temperature: float = 1,
            top_probability: float = 1,
            caching=None
            ) -> Generator[str, None, None]:
        self.llm.model_kwargs = {
            "temperature": temperature,
            "top_p": top_probability,
            "max_tokens_to_sample": 4000
            }
        yield from (item.text for item in self.llm._stream_messages(messages=messages))
