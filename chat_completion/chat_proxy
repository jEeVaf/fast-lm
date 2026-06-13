# proxy_server.py

@router.post("/v1/chat/completions")        # ← FastAPI route decorator
async def chat_completion(
    request: Request,
    fastapi_response: Response,
    model: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):

    # ════════════════════════════════════════════
    # STEP 1 — AUTH (already done by the time we
    # are inside this function body)
    # The Depends(user_api_key_auth) above does it
    # BEFORE this function body even starts running
    # ════════════════════════════════════════════

    data = await request.json()              # parse raw HTTP body → dict

    # ════════════════════════════════════════════
    # STEP 2 — GUARDRAILS / PRE-CALL HOOKS
    # ════════════════════════════════════════════
    await proxy_logging_obj.pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        data=data,
        call_type="completion"
    )

    # ════════════════════════════════════════════
    # STEP 3 — REQUEST ENRICHMENT
    # ════════════════════════════════════════════
    data["litellm_call_id"] = str(uuid.uuid4())
    data["metadata"] = {
        "user_api_key": user_api_key_dict.api_key,
        "team_id": user_api_key_dict.team_id,
    }
    start_time = datetime.now()

    # ════════════════════════════════════════════
    # ↓↓↓ THE HANDOFF LINE — RIGHT HERE ↓↓↓
    # ════════════════════════════════════════════

    if llm_router is not None and data["model"] in router_model_names:
        response = await llm_router.acompletion(**data)   # → router.py
    else:
        response = await litellm.acompletion(**data)      # → litellm directly

    # ════════════════════════════════════════════
    # AFTER the handoff — response handling
    # ════════════════════════════════════════════
    return response
