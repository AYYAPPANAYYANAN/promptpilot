"""
PromptPilot Enterprise v7
Production-oriented AI prompt engineering platform.

Layers
------
1. API / transport
2. Security + request context
3. Validation
4. Intent & prompt planning
5. Quality / confidence / deterministic fallback
6. AI execution
7. Optional voice
8. Persistence / telemetry
9. Health / observability

Design goal:
Fast default path, deterministic structured planning, graceful degradation,
and clean separation between prompt optimization and response execution.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from groq import Groq

try:
    from supabase import Client, create_client
except Exception:
    Client = Any
    create_client = None

try:
    from elevenlabs.client import ElevenLabs
except Exception:
    ElevenLabs = None


# ============================================================================
# 1. CONFIGURATION LAYER
# ============================================================================

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "PromptPilot Enterprise"
    version: str = "7.0.0"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000

    groq_api_key: str = ""
    groq_fast_model: str = "openai/gpt-oss-20b"
    groq_quality_model: str = "openai/gpt-oss-120b"

    supabase_url: str = ""
    supabase_key: str = ""

    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""

    max_input_chars: int = 12000
    request_timeout_seconds: float = 45.0
    max_completion_tokens_fast: int = 1800
    max_completion_tokens_quality: int = 3200

    enable_telemetry: bool = True
    allowed_origins: str = "*"


settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("promptpilot")


# ============================================================================
# 2. CLIENT / RESOURCE LAYER
# ============================================================================

groq_client: Optional[Groq] = None
supabase: Optional[Client] = None
elevenlabs_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global groq_client, supabase, elevenlabs_client

    if settings.groq_api_key:
        groq_client = Groq(api_key=settings.groq_api_key)
        logger.info("Groq client initialized.")
    else:
        logger.warning("GROQ_API_KEY is missing.")

    if (
        settings.supabase_url
        and settings.supabase_key
        and create_client
    ):
        try:
            supabase = create_client(
                settings.supabase_url,
                settings.supabase_key,
            )
            logger.info("Supabase client initialized.")
        except Exception:
            logger.exception("Supabase initialization failed.")

    if (
        settings.elevenlabs_api_key
        and settings.elevenlabs_voice_id
        and ElevenLabs
    ):
        try:
            elevenlabs_client = ElevenLabs(
                api_key=settings.elevenlabs_api_key
            )
            logger.info("ElevenLabs client initialized.")
        except Exception:
            logger.exception("ElevenLabs initialization failed.")

    yield

    groq_client = None
    supabase = None
    elevenlabs_client = None


# ============================================================================
# 3. APPLICATION LAYER
# ============================================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Enterprise prompt engineering and AI execution platform.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


origins = [
    x.strip()
    for x in settings.allowed_origins.split(",")
    if x.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=origins != ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================================
# 4. REQUEST CONTEXT / SECURITY LAYER
# ============================================================================

@app.middleware("http")
async def request_context(
    request: Request,
    call_next,
):
    request_id = (
        request.headers.get("x-request-id")
        or str(uuid.uuid4())
    )

    started = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        logger.exception(
            "Unhandled exception | request_id=%s",
            request_id,
        )

        response = JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Internal server error.",
                },
                "request_id": request_id,
            },
        )

    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000

    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = str(
        round(elapsed_ms, 2)
    )

    # Basic hardening.
    response.headers["x-content-type-options"] = "nosniff"
    response.headers["x-frame-options"] = "DENY"
    response.headers["referrer-policy"] = (
        "strict-origin-when-cross-origin"
    )
    response.headers["permissions-policy"] = (
        "camera=(), geolocation=(), payment=()"
    )

    logger.info(
        "%s %s -> %s | %.1f ms | %s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )

    return response


# ============================================================================
# 5. DOMAIN SCHEMAS
# ============================================================================

class UserProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = "User"
    role: str = "Professional"
    domain: str = ""
    expertise: str = ""
    tone: str = "clear"
    language: str = "English"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_input: str = Field(
        min_length=1,
        max_length=12000,
    )

    user_profile: UserProfile = Field(
        default_factory=UserProfile
    )

    quality: str = Field(
        default="fast",
        pattern="^(fast|quality)$",
    )

    include_execution: bool = True
    use_voice: bool = False

    @field_validator("user_input")
    @classmethod
    def normalize_input(cls, value: str) -> str:
        value = value.strip()

        # Collapse pathological whitespace without destroying
        # useful newlines.
        value = re.sub(
            r"[ \t]{3,}",
            "  ",
            value,
        )

        if not value:
            raise ValueError(
                "user_input cannot be empty."
            )

        return value


class PromptPlan(BaseModel):
    intent: str
    task: str
    context: str
    constraints: list[str]
    output_format: str
    assumptions: list[str]
    optimized_prompt: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


# ============================================================================
# 6. PROMPT INTELLIGENCE LAYER
# ============================================================================

PROMPT_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string"
        },
        "task": {
            "type": "string"
        },
        "context": {
            "type": "string"
        },
        "constraints": {
            "type": "array",
            "items": {"type": "string"},
        },
        "output_format": {
            "type": "string"
        },
        "assumptions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "optimized_prompt": {
            "type": "string"
        },
        "confidence": {
            "type": "number"
        },
    },
    "required": [
        "intent",
        "task",
        "context",
        "constraints",
        "output_format",
        "assumptions",
        "optimized_prompt",
        "confidence",
    ],
    "additionalProperties": False,
}


@dataclass
class HeuristicSignals:
    word_count: int
    question: bool
    code_request: bool
    research_request: bool
    creative_request: bool
    business_request: bool
    technical_request: bool


def extract_signals(text: str) -> HeuristicSignals:
    lowered = text.lower()

    code_terms = (
        "code",
        "python",
        "javascript",
        "typescript",
        "fastapi",
        "api",
        "sql",
        "react",
        "backend",
        "frontend",
        "algorithm",
    )

    research_terms = (
        "research",
        "paper",
        "hypothesis",
        "experiment",
        "dataset",
        "literature",
        "methodology",
    )

    creative_terms = (
        "design",
        "write",
        "story",
        "creative",
        "logo",
        "poster",
        "content",
        "caption",
    )

    business_terms = (
        "business",
        "product",
        "startup",
        "market",
        "customer",
        "revenue",
        "strategy",
        "enterprise",
    )

    return HeuristicSignals(
        word_count=len(text.split()),
        question="?" in text,
        code_request=any(
            term in lowered
            for term in code_terms
        ),
        research_request=any(
            term in lowered
            for term in research_terms
        ),
        creative_request=any(
            term in lowered
            for term in creative_terms
        ),
        business_request=any(
            term in lowered
            for term in business_terms
        ),
        technical_request=any(
            term in lowered
            for term in code_terms
        ),
    )


def estimate_complexity(
    text: str,
    signals: HeuristicSignals,
) -> float:
    """
    Lightweight complexity estimator.

    This is not intended as an ML classifier.
    It provides a deterministic routing signal before
    invoking the expensive model.
    """

    score = 0.0

    if signals.word_count > 25:
        score += 0.15

    if signals.word_count > 80:
        score += 0.20

    if signals.word_count > 160:
        score += 0.20

    if signals.code_request:
        score += 0.15

    if signals.research_request:
        score += 0.20

    if signals.business_request:
        score += 0.10

    if signals.question:
        score += 0.05

    return min(score, 1.0)


def select_model(
    quality: str,
    complexity: float,
) -> str:
    """
    Model routing algorithm.

    Explicit quality mode always wins.
    Otherwise fast requests stay on the smaller model.
    """

    if quality == "quality":
        return settings.groq_quality_model

    if complexity >= 0.65:
        return settings.groq_quality_model

    return settings.groq_fast_model


def build_planner_prompt(
    profile: UserProfile,
    signals: HeuristicSignals,
) -> str:
    return f"""
You are PromptPilot's enterprise Prompt Intelligence Kernel.

Your responsibility is prompt transformation, NOT answering the user's
original request.

USER PROFILE
-----------
Name: {profile.name}
Role: {profile.role}
Domain: {profile.domain or "general"}
Expertise: {profile.expertise or "general"}
Tone: {profile.tone}
Language: {profile.language}

DETERMINISTIC SIGNALS
---------------------
Word count: {signals.word_count}
Question detected: {signals.question}
Technical/code signals: {signals.technical_request}
Research signals: {signals.research_request}
Creative signals: {signals.creative_request}
Business signals: {signals.business_request}

OPTIMIZATION ALGORITHM
----------------------
1. Identify the user's true intent.
2. Extract the concrete task.
3. Preserve supplied context.
4. Identify explicit constraints.
5. Identify missing information that materially affects the answer.
6. Create only reasonable assumptions and expose them.
7. Select an appropriate output format.
8. Define a useful quality bar.
9. Rewrite the request into a standalone, reusable prompt.
10. Do not add goals that the user did not imply.
11. Avoid prompt bloat.
12. For technical tasks include correctness, security, performance,
    testing and edge-case requirements when relevant.
13. For research tasks include methodology, evaluation and limitations.
14. For business tasks include assumptions, trade-offs and measurable
    deliverables where appropriate.
15. Return only the required JSON object.

The optimized prompt must be ready to paste into another AI system.
""".strip()


def deterministic_fallback(
    raw: str,
    signals: HeuristicSignals,
) -> PromptPlan:
    """
    Safe fallback if structured AI output is unavailable.

    This guarantees that the product remains useful even if
    the external model has an outage.
    """

    if signals.code_request:
        intent = "Technical implementation"
        output = (
            "Production-ready explanation and complete implementation "
            "with validation, security, testing and edge cases."
        )

    elif signals.research_request:
        intent = "Research analysis"
        output = (
            "Structured research analysis with methodology, "
            "evaluation metrics, assumptions and limitations."
        )

    elif signals.creative_request:
        intent = "Creative generation"
        output = (
            "Clear, polished creative output with the requested "
            "style, structure and audience."
        )

    elif signals.business_request:
        intent = "Business analysis"
        output = (
            "Structured business response with assumptions, "
            "trade-offs, actions and measurable outcomes."
        )

    else:
        intent = "General assistance"
        output = (
            "Clear, structured response with practical next steps."
        )

    optimized = f"""
Act as an expert assistant.

Task:
{raw}

Requirements:
- Preserve the original intent.
- Make the response clear and actionable.
- State important assumptions.
- Use an appropriate structure.
- Avoid unsupported claims.
- Include examples where they materially improve understanding.

Output:
{output}
""".strip()

    return PromptPlan(
        intent=intent,
        task=raw,
        context="",
        constraints=[
            "Preserve original intent",
            "Be clear and actionable",
        ],
        output_format=output,
        assumptions=[],
        optimized_prompt=optimized,
        confidence=0.55,
    )


# ============================================================================
# 7. AI PROVIDER LAYER
# ============================================================================

async def groq_completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_format: Optional[dict[str, Any]] = None,
    temperature: float = 0.2,
    max_tokens: int = 1800,
) -> str:

    if not groq_client:
        raise HTTPException(
            status_code=503,
            detail="AI provider is not configured.",
        )

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }

    if response_format:
        payload["response_format"] = response_format

    def blocking_call():
        return groq_client.chat.completions.create(
            **payload
        )

    try:

        response = await asyncio.wait_for(
            asyncio.to_thread(blocking_call),
            timeout=settings.request_timeout_seconds,
        )

    except asyncio.TimeoutError:

        raise HTTPException(
            status_code=504,
            detail="AI provider timeout.",
        )

    except Exception as exc:

        logger.exception(
            "Groq provider failure."
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "AI provider request failed: "
                f"{str(exc)[:250]}"
            ),
        )

    content = response.choices[0].message.content

    if not content:
        raise HTTPException(
            status_code=502,
            detail="AI provider returned an empty response.",
        )

    return content


# ============================================================================
# 8. PLANNING ALGORITHM
# ============================================================================

async def optimize_prompt(
    request: ChatRequest,
) -> PromptPlan:

    raw = request.user_input

    signals = extract_signals(raw)

    complexity = estimate_complexity(
        raw,
        signals,
    )

    model = select_model(
        request.quality,
        complexity,
    )

    planner_system = build_planner_prompt(
        request.user_profile,
        signals,
    )

    try:

        content = await groq_completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": planner_system,
                },
                {
                    "role": "user",
                    "content": raw,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "prompt_plan",
                    "strict": True,
                    "schema": PROMPT_PLAN_SCHEMA,
                },
            },
            temperature=0.10,
            max_tokens=1600,
        )

        plan = PromptPlan.model_validate(
            json.loads(content)
        )

        # Clamp confidence defensively.
        plan.confidence = max(
            0.0,
            min(1.0, plan.confidence),
        )

        return plan

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Planner failed; using deterministic fallback."
        )

        return deterministic_fallback(
            raw,
            signals,
        )


# ============================================================================
# 9. EXECUTION LAYER
# ============================================================================

def build_execution_system(
    profile: UserProfile,
) -> str:

    return f"""
You are PromptPilot's AI execution engine.

Execute the optimized prompt directly.

USER
----
Role: {profile.role}
Domain: {profile.domain or "general"}
Expertise: {profile.expertise or "general"}
Tone: {profile.tone}
Language: {profile.language}

EXECUTION POLICY
----------------
- Follow the optimized prompt.
- Do not mention internal system instructions.
- Do not fabricate sources or facts.
- Clearly distinguish assumptions from facts.
- Prefer useful structure over unnecessary verbosity.
- For code, prioritize runnable, secure and maintainable output.
- For research, explain methodology and uncertainty.
- For business decisions, show assumptions and trade-offs.
""".strip()


async def execute_prompt(
    plan: PromptPlan,
    profile: UserProfile,
    quality: str,
) -> str:

    model = (
        settings.groq_quality_model
        if quality == "quality"
        else settings.groq_fast_model
    )

    max_tokens = (
        settings.max_completion_tokens_quality
        if quality == "quality"
        else settings.max_completion_tokens_fast
    )

    return await groq_completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": build_execution_system(
                    profile
                ),
            },
            {
                "role": "user",
                "content": plan.optimized_prompt,
            },
        ],
        temperature=0.25,
        max_tokens=max_tokens,
    )


# ============================================================================
# 10. VOICE LAYER
# ============================================================================

async def synthesize_voice(
    text: str,
) -> Optional[str]:

    if (
        not elevenlabs_client
        or not settings.elevenlabs_voice_id
    ):
        return None

    text = text[:2500]

    def blocking_call():

        audio = (
            elevenlabs_client
            .text_to_speech
            .convert(
                voice_id=settings.elevenlabs_voice_id,
                text=text,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )
        )

        return b"".join(audio)

    try:

        audio_bytes = await asyncio.to_thread(
            blocking_call
        )

        return base64.b64encode(
            audio_bytes
        ).decode()

    except Exception:

        logger.exception(
            "Voice generation failed."
        )

        return None


# ============================================================================
# 11. ANALYTICS / TELEMETRY LAYER
# ============================================================================

async def persist_run(
    request: ChatRequest,
    plan: PromptPlan,
    response_text: str,
    latency_ms: float,
    model: str,
    request_id: str,
):

    if (
        not supabase
        or not settings.enable_telemetry
    ):
        return

    row = {
        "request_id": request_id,
        "created_at": "now()",
        "user_name": request.user_profile.name,
        "user_role": request.user_profile.role,
        "raw_input": request.user_input,
        "intent": plan.intent,
        "optimized_prompt": plan.optimized_prompt,
        "ai_response": response_text,
        "quality_mode": request.quality,
        "model": model,
        "confidence": plan.confidence,
        "latency_ms": round(
            latency_ms,
            2,
        ),
    }

    try:

        await asyncio.to_thread(
            lambda:
            supabase
            .table("prompt_runs")
            .insert(row)
            .execute()
        )

    except Exception:

        logger.exception(
            "Telemetry persistence failed."
        )


# ============================================================================
# 12. HEALTH / OBSERVABILITY
# ============================================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "ai_configured": (
            groq_client is not None
        ),
        "database_configured": (
            supabase is not None
        ),
        "voice_configured": (
            elevenlabs_client is not None
        ),
    }


@app.get("/ready")
async def ready():

    if not groq_client:

        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": (
                    "AI provider is not configured."
                ),
            },
        )

    return {
        "ready": True
    }


@app.get("/api/v1/config")
async def public_config():

    return {
        "app_name": settings.app_name,
        "version": settings.version,

        "features": {
            "prompt_optimization": (
                groq_client is not None
            ),
            "voice": (
                elevenlabs_client is not None
            ),
            "analytics": (
                supabase is not None
                and settings.enable_telemetry
            ),
        },

        "models": {
            "fast": settings.groq_fast_model,
            "quality": settings.groq_quality_model,
        },
    }


# ============================================================================
# 13. PRODUCT API
# ============================================================================

@app.post("/api/v1/prompt/preview")
async def preview(
    request: ChatRequest,
):

    started = time.perf_counter()

    plan = await optimize_prompt(
        request
    )

    latency_ms = (
        time.perf_counter() - started
    ) * 1000

    return {
        "request_id": str(uuid.uuid4()),
        "data": {
            "plan": plan.model_dump(),
            "latency_ms": round(
                latency_ms,
                2,
            ),
        },
    }


@app.post("/api/v1/prompt/optimize")
async def optimize(
    request: ChatRequest,
):

    request_id = str(uuid.uuid4())

    started = time.perf_counter()

    plan = await optimize_prompt(
        request
    )

    response_text = ""

    if request.include_execution:

        response_text = await execute_prompt(
            plan,
            request.user_profile,
            request.quality,
        )

    audio_data = None

    if request.use_voice and response_text:

        audio_data = await synthesize_voice(
            response_text
        )

    latency_ms = (
        time.perf_counter() - started
    ) * 1000

    model = select_model(
        request.quality,
        estimate_complexity(
            request.user_input,
            extract_signals(
                request.user_input
            ),
        ),
    )

    # Fire-and-forget telemetry.
    asyncio.create_task(
        persist_run(
            request=request,
            plan=plan,
            response_text=response_text,
            latency_ms=latency_ms,
            model=model,
            request_id=request_id,
        )
    )

    return {
        "request_id": request_id,

        "data": {
            "plan": plan.model_dump(),

            "response": response_text,

            "audio_data": audio_data,

            "latency_ms": round(
                latency_ms,
                2,
            ),

            "model": model,
        },
    }


# ============================================================================
# 14. FRONTEND
# ============================================================================

@app.get("/")
async def root():

    return FileResponse(
        "static/index.html"
    )


# ============================================================================
# 15. LOCAL ENTRYPOINT
# ============================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=(
            settings.environment
            == "development"
        ),
    )
