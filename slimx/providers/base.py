from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Iterable, Optional, Sequence
from ..tooling import ToolSpec
from ..low.types import ChatRequest, ImageEditRequest, ImageRequest
from ..types import InspectedRequest, Result, StreamEvent

@dataclass(frozen=True)
class ProviderCapabilities:
    tools: bool = False
    structured_output: bool = False
    streaming: bool = False
    async_chat: bool = False
    async_streaming: bool = False
    # Multimodal: each flag must be backed by real serialization behavior.
    vision: bool = False        # image input
    documents: bool = False     # document (e.g. PDF) input
    audio_in: bool = False      # audio input
    image_out: bool = False     # image-generation output
    image_edit: bool = False    # image editing (edit_image / hosted edit action)
    hosted_image_tool: bool = False      # in-conversation image_generation tool
    image_partial_streaming: bool = False  # partial-image stream events

    @property
    def image_in(self) -> bool:
        """Alias for ``vision`` (image input), for symmetry with image_out/edit."""
        return self.vision


class Provider(ABC):
    name: str
    capabilities: ProviderCapabilities = ProviderCapabilities()

    @abstractmethod
    def chat(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec]=(),
        timeout: Optional[float]=None,
    ) -> Result: ...

    @abstractmethod
    def stream(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec]=(),
        timeout: Optional[float]=None,
    ) -> Iterable[StreamEvent]: ...

    # NOTE: These are intentionally *not* declared as `async def` so static
    # type checkers treat them as returning an awaitable/async-iterator.
    # Implementations may use `async def` / async generators.
    def achat(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec]=(),
        timeout: Optional[float]=None,
    ) -> Awaitable[Result]:
        raise NotImplementedError("Async not implemented for this provider")

    def astream(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec]=(),
        timeout: Optional[float]=None,
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError("Async streaming not implemented for this provider")

    # Dry-run inspection: build the exact HTTP request without sending it.
    # Optional; providers that support it return an InspectedRequest.
    def build_request(
        self,
        req: ChatRequest,
        *,
        tools: Sequence[ToolSpec]=(),
        stream: bool=False,
    ) -> InspectedRequest:
        raise NotImplementedError("Request inspection not implemented for this provider")

    # Model discovery: list model ids/names the provider/server exposes.
    # Optional; makes a network call. Providers that support it return a list of str.
    def list_models(self, *, timeout: Optional[float]=None) -> list:
        raise NotImplementedError("Model discovery not implemented for this provider")

    # Image generation: produce image(s) from a text prompt. Only providers that
    # declare `capabilities.image_out` implement these; the result carries the
    # images on `Result.images`.
    def generate_image(self, req: ImageRequest, *, timeout: Optional[float]=None) -> Result:
        raise NotImplementedError("Image generation not implemented for this provider")

    def agenerate_image(
        self, req: ImageRequest, *, timeout: Optional[float]=None
    ) -> Awaitable[Result]:
        raise NotImplementedError("Async image generation not implemented for this provider")

    # Image editing: refine source image(s) with an instruction. Only providers
    # that declare `capabilities.image_edit` implement these; edited images land
    # on `Result.images` just like generation.
    def edit_image(self, req: ImageEditRequest, *, timeout: Optional[float]=None) -> Result:
        raise NotImplementedError("Image editing not implemented for this provider")

    def aedit_image(
        self, req: ImageEditRequest, *, timeout: Optional[float]=None
    ) -> Awaitable[Result]:
        raise NotImplementedError("Async image editing not implemented for this provider")

    # Dry-run inspection for image generation (optional).
    def build_image_request(self, req: ImageRequest) -> InspectedRequest:
        raise NotImplementedError("Image-request inspection not implemented for this provider")
