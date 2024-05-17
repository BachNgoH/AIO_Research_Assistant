"""Google's hosted Gemini API."""

import os
import typing
from typing import Any, Dict, Optional, Sequence, Type, List
from llama_index.core.agent import AgentChatResponse
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_NUM_OUTPUTS, DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from llama_index.core.tools import FunctionTool
from llama_index.core.utilities.gemini_utils import (
    ROLES_FROM_GEMINI,
    merge_neighboring_same_role_messages,
)
from llama_index.llms.gemini.utils import (
    chat_from_gemini_response,
    chat_message_to_gemini,
    completion_from_gemini_response,
)

if typing.TYPE_CHECKING:
    import google.generativeai as genai


GEMINI_MODELS = (
    "models/gemini-pro",
    "models/gemini-ultra",
)


class GeminiForFunctionCalling(CustomLLM):
    """Gemini LLM.

    Examples:
        `pip install llama-index-llms-gemini`

        ```python
        from llama_index.llms.gemini import Gemini

        llm = Gemini(model_name="models/gemini-ultra", api_key="YOUR_API_KEY")
        resp = llm.complete("Write a poem about a magic backpack")
        print(resp)
        ```
    """

    model_name: str = Field(
        default=GEMINI_MODELS[0], description="The Gemini model to use."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        gte=0.0,
        lte=1.0,
    )
    max_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The number of tokens to generate.",
        gt=0,
    )
    generate_kwargs: dict = Field(
        default_factory=dict, description="Kwargs for generation."
    )
    
    # _memory: Sequence[ChatMessage] = []

    _model: "genai.GenerativeModel" = PrivateAttr()
    _model_meta: "genai.types.Model" = PrivateAttr()

    def __init__(
        self,
        tools: Optional[FunctionTool] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = GEMINI_MODELS[0],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
        generation_config: Optional["genai.types.GenerationConfigDict"] = None,
        safety_settings: "genai.types.SafetySettingOptions" = None,
        callback_manager: Optional[CallbackManager] = None,
        api_base: Optional[str] = None,
        transport: Optional[str] = None,
        system_prompt: str = None,
        **generate_kwargs: Any,
    ):
        """Creates a new Gemini model interface."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ValueError(
                "Gemini is not installed. Please install it with "
                "`pip install 'google-generativeai>=0.3.0'`."
            )
        if tools:
            tools = [tool.fn for tool in tools]
        # API keys are optional. The API can be authorised via OAuth (detected
        # environmentally) or by the GOOGLE_API_KEY environment variable.
        config_params: Dict[str, Any] = {
            "api_key": api_key or os.getenv("GOOGLE_API_KEY"),
        }
        
        if system_prompt is not None:
            self.system_prompt = [ChatMessage(content=system_prompt, role="system")]
        
        chat_history = chat_history or []
        # self.memory = memory or memory_cls.from_defaults(chat_history)
        
        if api_base:
            config_params["client_options"] = {"api_endpoint": api_base}
        if transport:
            config_params["transport"] = transport
        # transport: A string, one of: [`rest`, `grpc`, `grpc_asyncio`].
        genai.configure(**config_params)

        base_gen_config = generation_config if generation_config else {}
        # Explicitly passed args take precedence over the generation_config.
        final_gen_config = {"temperature": temperature, **base_gen_config}

        self._model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=final_gen_config,
            safety_settings=safety_settings,
            tools=tools,
        )

        self._model_meta = genai.get_model(model_name)

        supported_methods = self._model_meta.supported_generation_methods
        if "generateContent" not in supported_methods:
            raise ValueError(
                f"Model {model_name} does not support content generation, only "
                f"{supported_methods}."
            )

        if not max_tokens:
            max_tokens = self._model_meta.output_token_limit
        else:
            max_tokens = min(max_tokens, self._model_meta.output_token_limit)

        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            generate_kwargs=generate_kwargs,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Gemini_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        total_tokens = self._model_meta.input_token_limit + self.max_tokens
        return LLMMetadata(
            context_window=total_tokens,
            num_output=self.max_tokens,
            model_name=self.model_name,
            is_chat_model=True,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        result = self._model.generate_content(prompt, **kwargs)
        return completion_from_gemini_response(result)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        it = self._model.generate_content(prompt, stream=True, **kwargs)
        yield from map(completion_from_gemini_response, it)

    # @llm_chat_callback()
    def chat(self, messages: str, **kwargs: Any) -> ChatResponse:
        
        messages = [ChatMessage(content=messages, role="user")]
        merged_messages = merge_neighboring_same_role_messages(messages)
        *history, next_msg = map(chat_message_to_gemini, merged_messages)
        chat = self._model.start_chat(history=history)
        response = chat.send_message(next_msg)
        return AgentChatResponse(response=str(chat_from_gemini_response(response))) 

    def stream_chat(
        self, messages: str, **kwargs: Any
    ) -> ChatResponseGen:
        # self.memory.put(ChatMessage(content=messages, role="user"))
        
        # messages = self.get_all_messages()
        messages = [ChatMessage(content=messages, role="user")]

        merged_messages = merge_neighboring_same_role_messages(messages)
        *history, next_msg = map(chat_message_to_gemini, merged_messages)
        chat = self._model.start_chat(history=history, enable_automatic_function_calling=True)
        response = chat.send_message(next_msg, stream=False)

        
        def gen() -> ChatResponseGen:
            content = ""
            for r in response:
                top_candidate = r.candidates[0]
                content_delta = top_candidate.content.parts[0].text
                role = ROLES_FROM_GEMINI[top_candidate.content.role]
                raw = {
                    **(type(top_candidate).to_dict(top_candidate)),
                    **(
                        type(response.prompt_feedback).to_dict(response.prompt_feedback)
                    ),
                }
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=raw,
                )

        return gen()

    def get_all_messages(self) -> List[ChatMessage]:
        return (
            self.system_prompt
            + self.memory.get()
        )