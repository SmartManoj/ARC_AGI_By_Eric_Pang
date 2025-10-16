import asyncio
import base64
import io
import json
import os
import re
import time
import typing as T
from datetime import timedelta

import google.generativeai as genai
import PIL.Image
from anthropic import AsyncAnthropic, RateLimitError
from devtools import debug
from google.generativeai import caching as gemini_caching
from openai import AsyncAzureOpenAI, AsyncOpenAI
from xai_sdk import AsyncClient
from xai_sdk.chat import user, assistant, system, image

from src import logfire
from src.logic import random_string
from src.models import Attempt, Model, ModelUsage

if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def remove_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def text_only_messages(messages: list[dict[str, T.Any]]) -> list[dict[str, T.Any]]:
    new_messages = []
    for message in messages:
        content_strs: list[str] = []
        if isinstance(message["content"], str):
            content_strs.append(message["content"])
        else:
            for content in message["content"]:
                if content["type"] == "text":
                    content_strs.append(content["text"])
        if content_strs:
            new_messages.append(
                {
                    "role": message["role"],
                    "content": "\n".join(content_strs),
                }
            )
    return new_messages


async def get_next_message_anthropic(
    anthropic_client: AsyncAnthropic,
    system_messages: list[dict[str, T.Any]],
    messages: list[dict[str, T.Any]],
    model: Model,
    temperature: float,
    retry_secs: int = 15,
    max_retries: int = 200,
    stream: bool = False,
) -> tuple[str, ModelUsage] | None:
    retry_count = 0
    while True:
        try:
            request_id = random_string()
            start = time.time()
            logfire.debug(f"[{request_id}] calling anthropic")
            
            if not stream:
                message = await anthropic_client.beta.prompt_caching.messages.create(
                    system=system_messages,
                    temperature=temperature,
                    max_tokens=8_192,
                    messages=messages,
                    model=model.value,
                    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
                    timeout=120,
                )
                took_ms = (time.time() - start) * 1000
                usage = ModelUsage(
                    cache_creation_input_tokens=message.usage.cache_creation_input_tokens,
                    cache_read_input_tokens=message.usage.cache_read_input_tokens,
                    input_tokens=message.usage.input_tokens,
                    output_tokens=message.usage.output_tokens,
                )
                final_content = message.content[-1].text
            else:
                final_content = ""
                usage = None
                async with anthropic_client.beta.prompt_caching.messages.stream(
                    system=system_messages,
                    temperature=temperature,
                    max_tokens=8_192,
                    messages=messages,
                    model=model.value,
                    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
                    timeout=120,
                ) as stream_response:
                    async for text in stream_response.text_stream:
                        print(text, end="", flush=True)
                        final_content += text
                    
                    final_message = await stream_response.get_final_message()
                    usage = ModelUsage(
                        cache_creation_input_tokens=final_message.usage.cache_creation_input_tokens,
                        cache_read_input_tokens=final_message.usage.cache_read_input_tokens,
                        input_tokens=final_message.usage.input_tokens,
                        output_tokens=final_message.usage.output_tokens,
                    )
                print()  # newline after streaming
                took_ms = (time.time() - start) * 1000
            
            logfire.debug(
                f"[{request_id}] got back anthropic, took {took_ms:.2f}, {usage}, cost_cents={Attempt.cost_cents_from_usage(model=model, usage=usage)}"
            )
            break  # Success, exit the loop
        except RateLimitError:
            logfire.debug(
                f"Rate limit error, retrying in 15 seconds ({retry_count}/{max_retries})..."
            )
            retry_count += 1
            if retry_count >= max_retries:
                # raise  # Re-raise the exception after max retries
                return None
            await asyncio.sleep(retry_secs)
        except Exception as e:
            if "invalid x-api-key" in str(e):
                return None
            logfire.debug(
                f"Other anthropic error: {str(e)}, retrying in {retry_secs} seconds ({retry_count}/{max_retries})..."
            )
            retry_count += 1
            if retry_count >= max_retries:
                # raise  # Re-raise the exception after max retries
                return None
            await asyncio.sleep(retry_secs)
    return final_content, usage


async def get_next_message_deepseek(
    *,
    deepseek_client: AsyncOpenAI,
    messages: list[dict[str, T.Any]],
    model: Model,
    temperature: float,
    retry_secs: int = 15,
    max_retries: int = 50,
    use_baseten: bool,
) -> tuple[str, ModelUsage] | None:
    retry_count = 0
    MAX_CONTEXT_LENGTH = 65536
    params = {
        "temperature": temperature,
        "max_tokens": 8192,
        "messages": messages,
        "model": model.value,
        "timeout": 600,
        # "stream": False,
    }
    b10_str = " b10" if use_baseten else ""
    if use_baseten:
        params["model"] = "deepseek"
        params["extra_body"] = {
            "baseten": {
                "model_id": os.environ["BASETEN_R1_MODEL_ID"],
            }
        }
        params["max_tokens"] = 30_000
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
    while True:
        try:
            request_id = random_string()
            start = time.time()
            logfire.debug(f"[{request_id}] calling deepseek{b10_str}...")
            if not params.get("stream", None):
                print("calling")
                message = await deepseek_client.chat.completions.create(**params)
                cached_tokens = message.usage.prompt_tokens_details.cached_tokens
                usage = ModelUsage(
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=cached_tokens,
                    input_tokens=message.usage.prompt_tokens - cached_tokens,
                    output_tokens=message.usage.completion_tokens,
                )
                final_content = message.choices[0].message.content
            else:
                response = await deepseek_client.chat.completions.create(**params)
                final_content = ""
                usage = None
                count = 0
                async for chunk in response:
                    # print(chunk)
                    count += 1
                    if count % 100 == 0:
                        logfire.debug(f"[{request_id}] got chunk {count}")
                    if len(chunk.choices):
                        if chunk.choices[0].delta.content:
                            final_content += chunk.choices[0].delta.content
                            # print(final_content)
                    else:
                        if details := chunk.usage.prompt_tokens_details:
                            cached_tokens = details.cached_tokens or 0
                        else:
                            cached_tokens = 0
                        usage = ModelUsage(
                            cache_creation_input_tokens=0,
                            cache_read_input_tokens=cached_tokens,
                            input_tokens=chunk.usage.prompt_tokens - cached_tokens,
                            output_tokens=chunk.usage.completion_tokens,
                        )
                final_content = remove_thinking(text=final_content).strip()
                print(final_content)
                # TODO should i parse out thinking tags? probably

            took_ms = (time.time() - start) * 1000

            logfire.debug(
                f"[{request_id}] got back deepseek{b10_str}, took {took_ms:.2f}, {usage}, cost_cents={Attempt.cost_cents_from_usage(model=model, usage=usage)}"
            )
            break  # Success, exit the loop
        except Exception as e:
            error_msg = str(e)
            # Try to extract prompt tokens from error message
            if "tokens (" in error_msg:
                try:
                    prompt_tokens = int(
                        error_msg.split("(")[1].split(" in the messages")[0]
                    )
                    max_completion_tokens = MAX_CONTEXT_LENGTH - prompt_tokens
                    if max_completion_tokens <= 0:
                        return None
                    params["max_tokens"] = min(8192, max_completion_tokens)
                except (IndexError, ValueError):
                    pass  # If parsing fails, continue with normal retry logic
                    # raise e

            logfire.debug(
                f"Other deepseek{b10_str} error: {error_msg}, retrying in {retry_count} seconds ({retry_count}/{max_retries})..."
            )
            retry_count += 1
            if retry_count >= max_retries:
                return None
            await asyncio.sleep(retry_secs)
    return final_content, usage


async def get_next_message_openai(
    openai_client: AsyncOpenAI,
    messages: list[dict[str, T.Any]],
    model: Model,
    temperature: float,
    retry_secs: int = 15,
    max_retries: int = 3,
    name: str = "openai",
    stream: bool = False,
) -> tuple[str, ModelUsage] | None:
    retry_count = 0
    extra_params = {}
    if model not in [Model.o3_mini, Model.o1_mini, Model.o1_preview]:
        extra_params["temperature"] = temperature
    while True:
        try:
            request_id = random_string()
            start = time.time()
            logfire.debug(f"[{request_id}] calling openai")
            print(f"[{request_id}] calling openai with model {model.value}")
            
            if not stream:
                message = await openai_client.chat.completions.create(
                    **extra_params,
                    max_completion_tokens=16384,
                    messages=messages,
                    model=model.value,
                )
                cached_tokens = message.usage.prompt_tokens_details.cached_tokens
                usage = ModelUsage(
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=cached_tokens,
                    input_tokens=message.usage.prompt_tokens - cached_tokens,
                    output_tokens=message.usage.completion_tokens,
                )
                final_content = message.choices[0].message.content
            else:
                response = await openai_client.chat.completions.create(
                    **extra_params,
                    max_completion_tokens=16384,
                    messages=messages,
                    model=model.value,
                    stream=True,
                    stream_options={"include_usage": True}
                )
                final_content = ""
                usage = None
                async for chunk in response:
                    if len(chunk.choices):
                        if chunk.choices[0].delta.content:
                            print(chunk.choices[0].delta.content, end="", flush=True)
                            final_content += chunk.choices[0].delta.content
                    else:
                        if chunk.usage:
                            cached_tokens = chunk.usage.prompt_tokens_details.cached_tokens if chunk.usage.prompt_tokens_details else 0
                            usage = ModelUsage(
                                cache_creation_input_tokens=0,
                                cache_read_input_tokens=cached_tokens,
                                input_tokens=chunk.usage.prompt_tokens - cached_tokens,
                                output_tokens=chunk.usage.completion_tokens,
                            )
                print()  # newline after streaming
            
            took_ms = (time.time() - start) * 1000
            logfire.debug(
                f"[{request_id}] got back {name}, took {took_ms:.2f}, {usage}, cost_cents={Attempt.cost_cents_from_usage(model=model, usage=usage)}"
            )
            print(
                f"[{request_id}] got back {name}, took {took_ms:.2f}, {usage}, cost_cents={Attempt.cost_cents_from_usage(model=model, usage=usage)}"
            )
            break  # Success, exit the loop
        except Exception as e:
            logfire.debug(
                f"Other {name} error: {str(e)}, retrying in {retry_count} seconds ({retry_count}/{max_retries})..."
            )
            print(
                f"Other {name} error: {str(e)}, retrying in {retry_count} seconds ({retry_count}/{max_retries})..."
            )
            retry_count += 1
            if retry_count >= max_retries:
                # raise  # Re-raise the exception after max retries
                return None
            await asyncio.sleep(retry_secs)
    return final_content, usage

async def get_next_message_xai(
    xai_client: AsyncClient,
    messages: list[dict[str, T.Any]],
    model: Model,
    temperature: float,
    retry_secs: int = 15,
    max_retries: int = 0,
    name: str = "xai",
    stream: bool = False,
) -> tuple[str, ModelUsage] | None:
    retry_count = 0
    extra_params = {}
    extra_params["temperature"] = temperature
    while True:
        try:
            request_id = random_string()
            start = time.time()
            logfire.debug(f"[{request_id}] calling {name}")
            print(f"[{request_id}] calling {name} with model {model.value}")
            chat = xai_client.chat.create(model=model.value, max_tokens=120000, stream=stream)

            print(f"[{request_id}] chat successfully created")
            
            # Convert messages to XAI format
            for msg in messages:
                if msg["role"] == "system":
                    role = system
                elif msg["role"] == "user":
                    role = user
                elif msg["role"] == "assistant":
                    role = assistant
                else:
                    raise ValueError(f"Invalid role: {msg['role']}")

                for content in msg["content"]:
                    if content["type"] == "text":
                        chat.append(role(content["text"]))
                    elif content["type"] == "image_url":
                        chat.append(role(image(content["image_url"]["url"])))
                    else:
                        raise ValueError(f"Invalid content type: {content['type']}")

            logfire.debug(f"[{request_id}] chat: {chat}")
            
            if not stream:
                message = await chat.sample()
                final_content = message.content
                print(f"[{request_id}] message: {final_content}")
                logfire.debug(f"[{request_id}] message: {final_content}")
                cached_tokens = message.usage.cached_prompt_text_tokens
                usage = ModelUsage(
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=cached_tokens,
                    input_tokens=message.usage.prompt_tokens - cached_tokens,
                    output_tokens=message.usage.completion_tokens,
                )
            else:
                final_content = ""
                usage = None
                async for chunk in chat.sample_streaming():
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                        final_content += chunk.content
                    if chunk.usage:
                        cached_tokens = chunk.usage.cached_prompt_text_tokens
                        usage = ModelUsage(
                            cache_creation_input_tokens=0,
                            cache_read_input_tokens=cached_tokens,
                            input_tokens=chunk.usage.prompt_tokens - cached_tokens,
                            output_tokens=chunk.usage.completion_tokens,
                        )
                print()  # newline after streaming

            took_ms = (time.time() - start) * 1000
            logfire.debug(
                f"[{request_id}] got back {name}, took {took_ms:.2f}, {usage}, cost_cents={Attempt.cost_cents_from_usage(model=model, usage=usage)}"
            )
            print(
                f"[{request_id}] got back {name}, took {took_ms:.2f}, {usage}, cost_cents={Attempt.cost_cents_from_usage(model=model, usage=usage)}"
            )
            break  # Success, exit the loop
        except Exception as e:
            logfire.debug(
                f"Other {name} error: {str(e)}, retrying in {retry_count} seconds ({retry_count}/{max_retries})..."
            )
            print(
                f"Other {name} error: {str(e)}, retrying in {retry_count} seconds ({retry_count}/{max_retries})..."
            )
            retry_count += 1
            if retry_count >= max_retries:
                # raise  # Re-raise the exception after max retries
                return None
            await asyncio.sleep(retry_secs)
    return final_content, usage

async def get_next_message_gemini(
    cache: gemini_caching.CachedContent,
    model: Model,
    temperature: float,
    retry_secs: int = 15,
    max_retries: int = 200,
) -> tuple[str, ModelUsage] | None:
    retry_count = 0
    while True:
        try:
            request_id = random_string()
            start = time.time()
            logfire.debug(f"[{request_id}] calling gemini")

            genai_model = genai.GenerativeModel.from_cached_content(
                cached_content=cache
            )

            response = await genai_model.generate_content_async(
                contents=[
                    genai.types.ContentDict(
                        role="user", parts=[genai.types.PartDict(text="Please answer.")]
                    )
                ],
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    # max_output_tokens=10_000,
                ),
            )

            took_ms = (time.time() - start) * 1000
            usage = ModelUsage(
                cache_creation_input_tokens=0,
                cache_read_input_tokens=response.usage_metadata.cached_content_token_count,
                input_tokens=response.usage_metadata.prompt_token_count
                - response.usage_metadata.cached_content_token_count,
                output_tokens=response.usage_metadata.candidates_token_count,
            )
            logfire.debug(
                f"[{request_id}] got back gemini, took {took_ms:.2f}, {usage}, cost_cents={Attempt.cost_cents_from_usage(model=model, usage=usage)}"
            )
            break  # Success, exit the loop
        except Exception as e:
            if "invalid x-api-key" in str(e):
                return None
            logfire.debug(
                f"Other gemini error: {str(e)}, retrying in {retry_secs} seconds ({retry_count}/{max_retries})..."
            )
            retry_count += 1
            if retry_count >= max_retries:
                # raise  # Re-raise the exception after max retries
                return None
            await asyncio.sleep(retry_secs)
    return response.text, usage


async def get_next_messages(
    *, messages: list[dict[str, T.Any]], model: Model, temperature: float, n_times: int
) -> list[tuple[str, ModelUsage]] | None:
    if n_times <= 0:
        return []
    
    # Check if streaming is enabled via environment variable
    stream_enabled = os.environ.get("STREAM_LLM", "0") == "1" and n_times == 1
    
    if model in [Model.claude_3_5_sonnet, Model.claude_3_5_haiku]:
        if model == Model.claude_3_5_haiku:
            messages = text_only_messages(messages)
        anthropic_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        if messages[0]["role"] == "system":
            system_messages = messages[0]["content"]
            messages = messages[1:]
        else:
            system_messages = []
        cache_control_count = 0
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                for content in message["content"]:
                    if content["type"] == "image_url":
                        content["type"] = "image"
                        content["source"] = {
                            "data": content["image_url"]["url"].replace(
                                "data:image/png;base64,", ""
                            ),
                            "media_type": "image/png",
                            "type": "base64",
                        }
                        del content["image_url"]
                    if "cache_control" in content:
                        cache_control_count = cache_control_count + 1
                        if cache_control_count >= 3:
                            del content["cache_control"]

        # remove all the caches except for on the last one
        if isinstance(messages[-1]["content"], str):
            messages[-1]["content"] = [
                {"type": "text", "text": messages[-1]["content"]}
            ]
        messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        n_messages = [
            await get_next_message_anthropic(
                anthropic_client=anthropic_client,
                system_messages=system_messages,
                messages=messages,
                model=model,
                temperature=temperature,
                stream=stream_enabled,
            ),
            *await asyncio.gather(
                *[
                    get_next_message_anthropic(
                        anthropic_client=anthropic_client,
                        system_messages=system_messages,
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        stream=False,
                    )
                    for _ in range(n_times - 1)
                ]
            ),
        ]
        # filter out the Nones
        return [m for m in n_messages if m]
    elif model in [
        Model.gpt_4o,
        Model.gpt_4o_mini,
        Model.gpt_5,
        Model.o1_mini,
        Model.o1_preview,
        Model.o3_mini,
    ]:
        openai_client = AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=1200, # 1200 seconds = 20 minutes
            max_retries=10,
        )
        if messages[0]["role"] == "system":
            messages[0]["role"] = "developer"
        if model in [Model.o1_mini, Model.o1_preview, Model.o3_mini]:
            messages = text_only_messages(messages=messages)

        n_messages = [
            await get_next_message_openai(
                openai_client=openai_client,
                messages=messages,
                model=model,
                temperature=temperature,
                stream=stream_enabled,
            ),
            *await asyncio.gather(
                *[
                    get_next_message_openai(
                        openai_client=openai_client,
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        stream=False,
                    )
                    for _ in range(n_times - 1)
                ]
            ),
        ]
        return [m for m in n_messages if m]
    elif model in [Model.deep_seek_r1, Model.baseten_deepseek_r1]:
        if model == Model.deep_seek_r1:
            deepseek_client = AsyncOpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            )
            use_baseten = False
        elif model == Model.baseten_deepseek_r1:
            deepseek_client = AsyncOpenAI(
                api_key=os.environ["BASETEN_API_KEY"],
                base_url="https://bridge.baseten.co/v1/direct",
            )
            use_baseten = True
        else:
            raise ValueError(f"Invalid model: {model}")
        messages = text_only_messages(messages)

        if model == Model.deep_seek_r1:
            n_messages = [
                await get_next_message_deepseek(
                    deepseek_client=deepseek_client,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    use_baseten=use_baseten,
                ),
                *await asyncio.gather(
                    *[
                        get_next_message_deepseek(
                            deepseek_client=deepseek_client,
                            messages=messages,
                            model=model,
                            temperature=temperature,
                            use_baseten=use_baseten,
                        )
                        for _ in range(n_times - 1)
                    ]
                ),
            ]
        elif model == Model.baseten_deepseek_r1:
            n_messages = await asyncio.gather(
                *[
                    get_next_message_deepseek(
                        deepseek_client=deepseek_client,
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        use_baseten=use_baseten,
                    )
                    for _ in range(n_times)
                ]
            )
        else:
            raise ValueError(f"Invalid model: {model}")
        # filter out the Nones
    elif model in [Model.grok_3, Model.grok_4]:
        xai_client = AsyncClient(
            api_key=os.environ["XAI_API_KEY"],
            timeout=3600, # 3600 seconds = 60 minutes
        )

        print("Created xai client")

        if stream_enabled:
            n_messages = [
                await get_next_message_xai(
                    xai_client=xai_client,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    stream=True,
                )
            ]
        else:
            n_messages = await asyncio.gather(
                *[
                    get_next_message_xai(
                        xai_client=xai_client,
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        stream=False,
                    )
                    for _ in range(n_times)
                ]
            )
        return [m for m in n_messages if m]
    elif model in [Model.gemini_1_5_pro]:
        if messages[0]["role"] == "system":
            system_messages = messages[0]["content"]
            messages = messages[1:]
        else:
            system_messages = []
        system_instruction = system_messages[0]["text"]
        gemini_contents: list[genai.types.ContentDict] = []
        for message in messages:
            if message["role"] == "assistant":
                role = "model"
            else:
                role = message["role"]
            # debug(message["content"])
            if type(message["content"]) is str:
                parts = [genai.types.PartDict(text=message["content"])]
            else:
                parts = []
                for c in message["content"]:
                    if c["type"] == "text":
                        parts.append(genai.types.PartDict(text=c["text"]))
                    elif c["type"] == "image_url":
                        image = PIL.Image.open(
                            io.BytesIO(
                                base64.b64decode(
                                    c["image_url"]["url"].replace(
                                        "data:image/png;base64,", ""
                                    )
                                )
                            )
                        )
                        if image.mode == "RGBA":
                            image = image.convert("RGB")
                        parts.append(image)
            gemini_contents.append(genai.types.ContentDict(role=role, parts=parts))

        cache = gemini_caching.CachedContent.create(
            model=model.value,
            display_name=f"{random_string(10)}-{n_times}",  # used to identify the cache
            system_instruction=system_instruction,
            contents=gemini_contents,
            ttl=timedelta(minutes=5),
        )

        n_messages = [
            *await asyncio.gather(
                *[
                    get_next_message_gemini(
                        cache=cache, model=model, temperature=temperature
                    )
                    for _ in range(n_times)
                ]
            ),
        ]
        # filter out the Nones
        return [m for m in n_messages if m]
    elif model in [Model.openrouter_claude_3_5_sonnet, Model.openrouter_model, Model.openrouter_o1, Model.openrouter_o1_mini]:
        openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        if model in [Model.openrouter_model, Model.openrouter_o1, Model.openrouter_o1_mini]:
            messages = text_only_messages(messages)
        
        attempt_counter = {"count": 0}
        attempt_lock = asyncio.Lock()
        
        async def get_message_with_retry(messages, model, temperature, max_retries=5):
            """Wrapper to retry get_next_message with exponential backoff"""
            # Get unique attempt number
            async with attempt_lock:
                attempt_counter["count"] += 1
                attempt_num = attempt_counter["count"]
            
            print(f"[Attempt {attempt_num}] Starting API call to {model.value}...")
            for retry in range(max_retries):
                try:
                    result = await get_next_message(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        attempt_num=attempt_num,
                    )
                    print(f"[Attempt {attempt_num}] ✓ Received response from {model.value}")
                    return result
                except Exception as e:
                    error_str = str(e).lower()
                    # Retry on rate limits (429), upstream errors (502), or empty choices
                    should_retry = (
                        "429" in str(e) or 
                        "rate" in error_str or 
                        "502" in str(e) or 
                        "upstream" in error_str or
                        "no choices" in error_str
                    )
                    if should_retry:
                        wait_time = 2 ** retry * 10  # 10, 20, 40, 80, 160 seconds
                        print(f"API error (retry {retry+1}/{max_retries}), waiting {wait_time}s: {str(e)[:100]}...")
                        logfire.debug(f"API error: {e}, retrying in {wait_time}s")
                        if retry < max_retries - 1:
                            await asyncio.sleep(wait_time)
                        else:
                            raise
                    else:
                        print(f"Non-retryable error: {e}")
                        raise
            return None
        
        n_messages = await asyncio.gather(
            *[
                get_message_with_retry(messages, model, temperature)
                for _ in range(n_times)
            ]
        )
        return [m for m in n_messages if m]
    else:
        raise ValueError(f"Invalid model: {model}")


async def get_next_message(
    *, messages: list[dict[str, T.Any]], model: Model, temperature: float, attempt_num: int = 0
) -> tuple[str, ModelUsage]:
    if int(os.environ.get("NO_WIFI", 0)) == 1:
        return "[[1, 2, 3], [4, 5, 6]]", ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            input_tokens=0,
            output_tokens=0,
        )
    if model in [Model.claude_3_5_sonnet, Model.claude_3_5_haiku]:
        anthropic_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        if messages[0]["role"] == "system":
            system_messages = messages[0]["content"]
            messages = messages[1:]
        else:
            system_messages = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                for content in message["content"]:
                    if content["type"] == "image_url":
                        content["type"] = "image"
                        content["source"] = {
                            "data": content["image_url"]["url"].replace(
                                "data:image/png;base64,", ""
                            ),
                            "media_type": "image/png",
                            "type": "base64",
                        }
                        del content["image_url"]

        retry_count = 0
        max_retries = 12
        while True:
            try:
                message = await anthropic_client.beta.prompt_caching.messages.create(
                    system=system_messages,
                    temperature=temperature,
                    max_tokens=8_192,
                    messages=messages,
                    model=model.value,
                    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
                    timeout=120,
                )
                break  # Success, exit the loop
            except RateLimitError:
                logfire.debug(
                    f"Rate limit error, retrying in 30 seconds ({retry_count}/{max_retries})..."
                )
                retry_count += 1
                if retry_count >= max_retries:
                    raise  # Re-raise the exception after max retries
                await asyncio.sleep(15)  # Wait for 30 seconds before retrying

        return message.content[-1].text, ModelUsage(
            cache_creation_input_tokens=message.usage.cache_creation_input_tokens,
            cache_read_input_tokens=message.usage.cache_read_input_tokens,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
    elif model in [Model.gpt_4o, Model.gpt_4o_mini]:
        openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        message = await openai_client.chat.completions.create(
            model=model.value,
            messages=messages,
            temperature=temperature,
            max_tokens=10_000,
        )
        cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.nvidia_llama_3_1_nemotron_70b_instruct:
        nvidia_client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
        message = await nvidia_client.chat.completions.create(
            model=model.value,
            messages=text_only_messages(messages),
            temperature=temperature,
            max_tokens=10_000,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.groq_llama_3_2_90b_vision:
        groq_client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )
        message = await groq_client.chat.completions.create(
            model=model.value,
            messages=text_only_messages(messages),
            temperature=temperature,
            max_tokens=8_192,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.openrouter_claude_3_5_sonnet:
        openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        message = await openrouter_client.chat.completions.create(
            model=model.value,
            messages=messages,
            temperature=temperature,
            max_tokens=10_000,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.openrouter_model:
        openrouter_client = AsyncOpenAI(
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("OPENROUTER_API_KEY", "dummy"),
        )
        
        # Check if streaming is enabled via environment variable
        stream_enabled = os.environ.get("STREAM_LLM", "0") == "1"
        
        if not stream_enabled:
            message = await openrouter_client.chat.completions.create(
                model=model.value,
                messages=messages,
                temperature=temperature,
                max_tokens=20_000,
            )
            if not message.choices or len(message.choices) == 0:
                # log messages to file
                with open("messages.json", "w") as f:
                    f.write(json.dumps(messages, indent=4))
                print(f"OpenRouter API error - no choices returned. Full response: {message};")
                raise ValueError(f"OpenRouter API returned no choices. Check API key, rate limits, or model availability.")
            if message.usage and message.usage.prompt_tokens_details:
                cached_tokens = message.usage.prompt_tokens_details.cached_tokens
            else:
                cached_tokens = 0
            final_content = message.choices[0].message.content
            usage = ModelUsage(
                cache_creation_input_tokens=0,
                cache_read_input_tokens=cached_tokens,
                input_tokens=message.usage.prompt_tokens - cached_tokens,
                output_tokens=message.usage.completion_tokens,
            )
        else:
            # Create a unique file for this streaming attempt
            from pathlib import Path
            stream_dir = Path("stream_outputs")
            stream_dir.mkdir(exist_ok=True)
            stream_file = stream_dir / f"attempt_{attempt_num}_{random_string(4)}.txt"
            
            response = await openrouter_client.chat.completions.create(
                model=model.value,
                messages=messages,
                temperature=temperature,
                max_tokens=20_000,
                stream=True,
                stream_options={"include_usage": True}
            )
            final_content = ""
            usage = None
            
            print(f"[Attempt {attempt_num}] Streaming to {stream_file}")
            
            with open(stream_file, "w", encoding="utf-8") as f:
                async for chunk in response:
                    if len(chunk.choices) > 0:
                        if chunk.choices[0].delta.content:
                            content_chunk = chunk.choices[0].delta.content
                            f.write(content_chunk)
                            f.flush()
                            final_content += content_chunk
                    else:
                        if chunk.usage:
                            if chunk.usage.prompt_tokens_details:
                                cached_tokens = chunk.usage.prompt_tokens_details.cached_tokens
                            else:
                                cached_tokens = 0
                            usage = ModelUsage(
                                cache_creation_input_tokens=0,
                                cache_read_input_tokens=cached_tokens,
                                input_tokens=chunk.usage.prompt_tokens - cached_tokens,
                                output_tokens=chunk.usage.completion_tokens,
                            )
            
            print(f"[Attempt {attempt_num}] Streaming complete, saved to {stream_file}")
            
            if not final_content:
                print(f"[Attempt {attempt_num}] OpenRouter API error - no content received")
                raise ValueError(f"OpenRouter API returned no content. Check API key, rate limits, or model availability.")
        
        return final_content, usage
    elif model == Model.openrouter_o1_mini:
        openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        message = await openrouter_client.chat.completions.create(
            model=model.value,
            messages=messages,
            temperature=temperature,
            max_tokens=20_000,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == [Model.azure_gpt_4o, Model.azure_gpt_4o_mini]:
        azure_client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-10-01-preview",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
        message = await azure_client.chat.completions.create(
            model=model.value.replace("azure-", ""),
            messages=messages,
            temperature=temperature,
            max_tokens=10_000,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.gemini_1_5_pro:
        if messages[0]["role"] == "system":
            system_messages = messages[0]["content"]
            messages = messages[1:]
        else:
            system_messages = []
        model = genai.GenerativeModel(
            model.value, system_instruction=system_messages[0]["text"]
        )
        gemini_contents = []
        for message in messages:
            if message["role"] == "assistant":
                role = "model"
            else:
                role = message["role"]
            # debug(message["content"])
            if type(message["content"]) is str:
                parts = [genai.types.PartDict(text=message["content"])]
            else:
                parts = []
                for c in message["content"]:
                    if c["type"] == "text":
                        parts.append(genai.types.PartDict(text=c["text"]))
                    elif c["type"] == "image_url":
                        image = PIL.Image.open(
                            io.BytesIO(
                                base64.b64decode(
                                    c["image_url"]["url"].replace(
                                        "data:image/png;base64,", ""
                                    )
                                )
                            )
                        )
                        if image.mode == "RGBA":
                            image = image.convert("RGB")
                        parts.append(image)
            gemini_contents.append(genai.types.ContentDict(role=role, parts=parts))
        response = await model.generate_content_async(
            contents=gemini_contents,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=10_000,
            ),
        )
        return response.text, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        )
    else:
        raise ValueError(f"Invalid model: {model}")


noop_code = """
def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    raise NotImplementedError()
""".strip()


def clean_code(s: str) -> str:
    return s.replace("\t", " " * 4)


def parse_python_backticks(s: str) -> str:
    if s.count("```python") == 0:
        logfire.debug("NO CODE BLOCKS")
        out = s.partition("</reasoning>")[2]
        if out == "":
            return noop_code
        return clean_code(out)

    if s.count("```python") > 1:
        # print(f"MULTIPLE CODE BLOCKS\n=====\n\n{s}\n\n=====")
        for chunk in s.split("```python")[::-1]:
            if "def transform(" in chunk:
                s = "```python" + chunk
                break

    assert s.count("```python") == 1

    attempted_search = re.search(r"```python\n(.*)\n```", s, re.DOTALL | re.MULTILINE)
    if attempted_search is not None:
        return clean_code(attempted_search.group(1))

    attempted_search = re.search(r"```python\n(.*)\n`", s, re.DOTALL | re.MULTILINE)
    if attempted_search is not None:
        logfire.debug("PARSE ERROR CASE (1)")
        return clean_code(attempted_search.group(1))
    else:
        logfire.debug("PARSE ERROR CASE (2!)")

    return clean_code(s.partition("```python")[2])


def parse_2d_arrays_from_string(s: str) -> list[list[list[int]]]:
    # Regular expression pattern to match 2D arrays
    pattern = r"\[\s*(\[[^\[\]]*\](?:,\s*\[[^\[\]]*\])*\s*)\]"

    # Find all matches of the pattern in the output string
    matches = re.findall(pattern, s)

    # Process each match to create a list of 2D arrays
    arrays_list: list[list[list[int]]] = []

    for match in matches:
        # Find all inner arrays within the matched 2D array
        rows = re.findall(r"\[([^\]]*)\]", match)
        array_2d = []
        for row in rows:
            # Split the row by commas and convert to integers
            nums = [int(n.strip()) for n in row.split(",") if n.strip()]
            array_2d.append(nums)
        arrays_list.append(array_2d)

    return arrays_list
