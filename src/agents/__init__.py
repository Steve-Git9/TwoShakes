"""
AgentClient: unified LLM wrapper for all DataPrepAgent agents.

Priority:
1. Microsoft Agent Framework  — azure.ai.projects.AIProjectClient (Azure AI Agents)
2. Azure OpenAI SDK fallback  — openai.AzureOpenAI

This is the ONLY place in the codebase that talks to the LLM.
"""

import os
import logging
import asyncio

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Tier 1: Microsoft Agent Framework (azure-ai-projects) ─────────────────────
_AGENT_FRAMEWORK_AVAILABLE = False
try:
    from azure.ai.projects import AIProjectClient          # type: ignore
    from azure.core.credentials import AzureKeyCredential  # type: ignore
    _AGENT_FRAMEWORK_AVAILABLE = True
    logger.info("Backend available: Microsoft Agent Framework (azure-ai-projects)")
except ImportError:
    logger.info("azure-ai-projects not importable — will use Azure OpenAI SDK fallback")


class AgentClient:
    """
    Unified LLM client used by all agents in the pipeline.

    Backend priority:
    1. Microsoft Agent Framework via azure.ai.projects.AIProjectClient
       (Azure AI Agents — the production Microsoft Agent Framework)
    2. Azure OpenAI SDK (openai.AzureOpenAI) pointed at the Foundry endpoint
    """

    def __init__(self, name: str, instructions: str, json_mode: bool = False):
        self.name = name
        self.json_mode = json_mode

        # Inject JSON reminder into instructions when json_mode=True so both
        # backends produce parseable output even when response_format isn't set.
        if json_mode:
            self.instructions = (
                instructions.rstrip()
                + "\n\nIMPORTANT: Respond ONLY with valid JSON — no markdown fences, "
                "no extra text."
            )
        else:
            self.instructions = instructions

        self._endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT", "")
        self._api_key  = os.getenv("AZURE_AI_PROJECT_KEY", "")
        self._deployment = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o-mini")

        if not self._endpoint or not self._api_key:
            raise ValueError(
                "AZURE_AI_PROJECT_ENDPOINT and AZURE_AI_PROJECT_KEY must be set in .env"
            )

        # Always initialise the OpenAI fallback client (used if AF fails)
        self._init_openai()

        # Determine active backend
        self._backend = "agent-framework" if _AGENT_FRAMEWORK_AVAILABLE else "openai-sdk"
        logger.info(f"[{self.name}] Initialized — backend: {self._backend}")

    # ── Initialise Azure OpenAI fallback ──────────────────────────────────────
    def _init_openai(self):
        from openai import AzureOpenAI  # type: ignore
        self._openai_client = AzureOpenAI(
            azure_endpoint=self._endpoint,
            api_key=self._api_key,
            api_version="2024-12-01-preview",
        )

    # ── Public interface ──────────────────────────────────────────────────────
    async def run(self, message: str) -> str:
        """Send a message to the LLM and return the text response."""
        if self._backend == "agent-framework":
            try:
                return await self._run_agent_framework(message)
            except Exception as e:
                logger.warning(
                    f"[{self.name}] Agent Framework call failed ({e}), "
                    "falling back to Azure OpenAI SDK"
                )
                self._backend = "openai-sdk"

        return await self._run_openai(message)

    # ── Backend: Microsoft Agent Framework ───────────────────────────────────
    async def _run_agent_framework(self, message: str) -> str:
        """
        Call the LLM via Azure AI Agents (Microsoft Agent Framework).

        Creates a disposable agent + thread per call so agents are stateless
        and can be used concurrently. Agent is deleted after the response is
        received to avoid accumulating stale agents in the project.
        """
        loop = asyncio.get_event_loop()

        instructions = self.instructions
        deployment   = self._deployment
        endpoint     = self._endpoint
        api_key      = self._api_key
        json_mode    = self.json_mode
        name         = self.name

        def _call_sync():
            project_client = AIProjectClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key),
            )

            # Build optional response_format for JSON mode
            create_kwargs: dict = dict(
                model=deployment,
                name=name,
                instructions=instructions,
            )
            if json_mode:
                create_kwargs["response_format"] = {"type": "json_object"}

            agent  = project_client.agents.create_agent(**create_kwargs)
            thread = project_client.agents.create_thread()

            project_client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=message,
            )

            # Blocks until the run finishes (polling handled internally)
            project_client.agents.create_and_process_run(
                thread_id=thread.id,
                agent_id=agent.id,
            )

            messages = project_client.agents.list_messages(thread_id=thread.id)
            reply    = messages.get_last_text_message_by_role("assistant").text.value

            # Clean up — avoid accumulating stale agents in the Foundry project
            try:
                project_client.agents.delete_agent(agent.id)
            except Exception:
                pass

            return reply

        return await loop.run_in_executor(None, _call_sync)

    # ── Backend: Azure OpenAI SDK (fallback) ─────────────────────────────────
    async def _run_openai(self, message: str) -> str:
        kwargs: dict = {
            "model": self._deployment,
            "messages": [
                {"role": "system", "content": self.instructions},
                {"role": "user",   "content": message},
            ],
            "temperature": 0.2,
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._openai_client.chat.completions.create(**kwargs),
        )
        return response.choices[0].message.content
