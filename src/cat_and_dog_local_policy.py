from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Union

import json
import re

from core import Message
from llm.request import LLMType, RequestOutput, create_client, get_response_from_llm
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextvars
from pathlib import Path
try:
    # Optional, but enables per-call tracing when available.
    from langfuse import observe  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    def observe(*_args, **_kwargs):  # type: ignore[no-redef]
        def _decorator(fn):
            return fn
        return _decorator

try:
    # Optional: lets us set a Langfuse session_id to group traces.
    from utils.tracing import langfuse_client  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    langfuse_client = None  # type: ignore[assignment]


ChatHistoryLike = Sequence[Union[Message, Mapping[str, Any]]]


def _normalize_messages(messages: ChatHistoryLike) -> tuple[str, List[Message]]:
    """
    Convert mixed message formats into (system_message, msg_history).
    """
    system_parts: List[str] = []
    msg_history: List[Message] = []

    for m in messages:
        if isinstance(m, Message):
            role = m.role
            content = m.content
        elif isinstance(m, Mapping):
            role = str(m.get("role", ""))
            content = str(m.get("content", ""))
        else:
            raise TypeError(f"Unsupported message type: {type(m)}")

        if role == "system":
            if content.strip():
                system_parts.append(content)
        else:
            msg_history.append(Message(role=role, content=content))

    system_message = "\n\n".join(system_parts).strip()
    return system_message, msg_history


@observe(capture_input=False, capture_output=False)
def query_llm_many_seeds_with_chat_history(
    messages: ChatHistoryLike,
    *,
    n_seeds: int,
    n_queries_per_seed: int,
    llmtype: LLMType,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = "medium",
    base_seed: Optional[int] = 0,
    use_langfuse: bool = True,
    langfuse_name_prefix: str = "cat_and_dog_local_policy",
    langfuse_session_id: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> List[List[RequestOutput]]:
    """
    Query an LLM many times in parallel across multiple seeds.

    Returns a nested list `outputs[seed_idx][query_idx] = RequestOutput`.
    All requests are executed in a single thread pool and (when Langfuse is enabled)
    are recorded under the same parent trace/span created by this function.
    """
    if n_seeds <= 0:
        raise ValueError(f"n_seeds must be > 0, got {n_seeds}.")
    if n_queries_per_seed <= 0:
        raise ValueError(f"n_queries_per_seed must be > 0, got {n_queries_per_seed}.")
    if not isinstance(messages, Sequence) or len(messages) == 0:
        raise ValueError("messages must be a non-empty sequence.")

    system_message, msg_history = _normalize_messages(messages)

    # Ensure all calls are grouped under one Langfuse "session" (if requested).
    if use_langfuse and langfuse_session_id and langfuse_client is not None:
        langfuse_client.update_current_trace(session_id=langfuse_session_id)

    def _run_one(seed_idx: int, query_idx: int) -> RequestOutput:
        client, model = create_client(llmtype, use_langfuse=use_langfuse)
        req_seed = None
        if base_seed is not None:
            # Make per-(seed,query) distinct seeds.
            req_seed = int(base_seed) + int(seed_idx) * 10_000 + int(query_idx)

        # IMPORTANT: capture_input=False prevents leaking client/api_key into Langfuse.
        return observe(
            name=f"{langfuse_name_prefix}:seed{seed_idx}:q{query_idx}",
            capture_input=False,
            capture_output=False,
        )(get_response_from_llm)(
            client=client,
            model=model,
            msg_history=msg_history,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            seed=req_seed,
        )

    if max_workers is None:
        max_workers = min(16, n_seeds * n_queries_per_seed)
    max_workers = max(1, int(max_workers))

    outputs: List[List[Optional[RequestOutput]]] = [
        [None for _ in range(n_queries_per_seed)] for _ in range(n_seeds)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for seed_idx in range(n_seeds):
            for query_idx in range(n_queries_per_seed):
                # Run inside a *fresh* copied context per task to preserve tracing contextvars.
                # NOTE: a single Context instance cannot be entered concurrently across threads.
                ctx = contextvars.copy_context()
                fut = ex.submit(ctx.run, _run_one, seed_idx, query_idx)
                futures[fut] = (seed_idx, query_idx)

        for fut in as_completed(futures):
            seed_idx, query_idx = futures[fut]
            outputs[seed_idx][query_idx] = fut.result()

    # type narrowing
    return [[o for o in row if o is not None] for row in outputs]


@observe(capture_input=False, capture_output=False)
def query_llm_n_times_with_chat_history(
    messages: ChatHistoryLike,
    n_queries: int,
    llmtype: LLMType,
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = "medium",
    seed: Optional[int] = 42,
    use_langfuse: bool = True,
    langfuse_name_prefix: str = "cat_and_dog_local_policy",
    max_workers: Optional[int] = None,
) -> List[RequestOutput]:
    """
    Query an LLM `n_queries` times using the same chat history.

    Args:
        messages:
            Chat history as either:
            - list[core.Message]
            - list[{"role": str, "content": str}] (OpenAI-style dicts)
            Any messages with role=="system" are extracted and concatenated into a single
            system prompt (since `get_response_from_llm` takes `system_message` separately).
        n_queries: Number of independent calls to make.
        llmtype: Which model to query (see `llm.request.LLMType`).
        temperature, max_tokens, reasoning_effort: Passed through to the LLM call.
        seed:
            Base seed for deterministic sampling; each query uses `seed + i` to avoid
            returning identical outputs when the backend supports seeding.
            Set to None to let the backend choose.
        use_langfuse: Whether to use the langfuse-wrapped OpenAI client.

    Returns:
        List[RequestOutput] (one per query).
    """
    if n_queries <= 0:
        raise ValueError(f"n_queries must be > 0, got {n_queries}.")
    if not isinstance(messages, Sequence) or len(messages) == 0:
        raise ValueError("messages must be a non-empty sequence.")

    # Backwards-compatible wrapper: single "seed" and n_queries_per_seed=n_queries
    # but return a flat list.
    out = query_llm_many_seeds_with_chat_history(
        messages,
        n_seeds=1,
        n_queries_per_seed=n_queries,
        llmtype=llmtype,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        base_seed=seed,
        use_langfuse=use_langfuse,
        langfuse_name_prefix=langfuse_name_prefix,
        max_workers=max_workers,
    )
    return out[0]

def get_base_messages(env_name: str) -> ChatHistoryLike:
    messages = [
        {
            "role": "system",
            "content": ""
        }
    ]
    if env_name == "cat_and_dog":
        messages.append(
            {
                "role": "user",
                "content": "\nYou are playing a cooperative game with another player. This game is generally turn-based. While you cannot communicate with the other player, you can see their actions and the (joined) rewards from previous turns. Your goal is to maximize your (joint) reward. \nNote that you are only playing one round of the game (one round consists of multiple turns).\n\nNow follows a detailed description of the game and (if applicable) the history of previous turns.\n\n\nYou are playing a game of cat and dog with one other player. One of you is Alice and the other is Bob.\nAlice is separated from Bob by a wall. The goal of the game is maximize your (joint) reward. \nOn Alice's side of the wall, there is either a cat or a dog. \nAdditionally, there is a light switch on Alice's side of the wall, which she can turn on to illuminate a bulb on Bob's side of the wall.\n\nAlice goes first and can do exactly one of the following:\n- bail for a (joint) reward of 1 (this ends the game immediately)\n- remove the wall for a (joint) reward of -5\n- turn on the light bulb (no reward)\n- do nothing, i.e. the light bulb stays off (no reward)\n\nIf Alice didn't bail, it is Bob's turn. Bob observes the light status and the animal on Alice's side of the wall, if the wall was removed by Alice.\nBob can do exactly one of the following:\n- bail for a (joint) reward of 0.5\n- or guess the animal, i.e. either cat or dog\n    - if Bob guesses correctly, he receives a (joint) reward of 10\n    - if Bob guesses incorrectly, he receives a (joint) reward of -10\n\nThe players cannot communicate with each other except the options presented above (i.e. light status and removing the wall).\n"
            },
        )
    elif env_name == "cat_and_dog_semantics":
        messages.append(
            {
                "role": "user",
                "content": "\nYou are playing a cooperative game with another player. This game is generally turn-based. While you cannot communicate with the other player, you can see their actions and the (joined) rewards from previous turns. Your goal is to maximize your (joint) reward. \nNote that you are only playing one round of the game (one round consists of multiple turns).\n\nNow follows a detailed description of the game and (if applicable) the history of previous turns.\n\n\nYou are playing a game of cat and dog with one other player. One of you is Alice and the other is Bob.\nAlice is separated from Bob by a wall. The goal of the game is maximize your (joint) reward. \nOn Alice's side of the wall, there is either a cat or a dog. \nAdditionally, there is a light switch on Alice's side of the wall, which she can use to set a bulb on Bob's side of the wall to either a red light or a blue light.\n\nAlice goes first and can do exactly one of the following:\n- bail for a (joint) reward of 1 (this ends the game immediately)\n- remove the wall for a (joint) reward of -5\n- turn on the red light (no reward)\n- turn on the blue light (no reward)\n\nIf Alice didn't bail, it is Bob's turn. Bob observes the light color and the animal on Alice's side of the wall, if the wall was removed by Alice.\nBob can do exactly one of the following:\n- bail for a (joint) reward of 0.5\n- or guess the animal, i.e. either cat or dog\n    - if Bob guesses correctly, he receives a (joint) reward of 10\n    - if Bob guesses incorrectly, he receives a (joint) reward of -10\n\nThe players cannot communicate with each other except the options presented above (i.e. light color and removing the wall).\n"
            },
        )
    elif env_name == "cat_and_dog_german":
        messages.append(
            {
                "role": "user",
                "content": "\nDu spielst ein kooperatives Spiel mit einem anderen Spieler. Dieses Spiel ist im Allgemeinen rundenbasiert. Obwohl du nicht mit dem anderen Spieler kommunizieren kannst, kannst du seine Aktionen sowie die (gemeinsamen) Belohnungen aus vorherigen Zügen sehen. Dein Ziel ist es, eure (gemeinsame) Belohnung zu maximieren. \nBeachte, dass ihr nur eine Runde des Spiels spielt (eine Runde besteht aus mehreren Zügen).\n\nNun folgt eine detaillierte Beschreibung des Spiels und (falls zutreffend) die Historie der vorherigen Züge.\n\n\nDu spielst ein Spiel von Katze und Hund mit einem weiteren Spieler. Einer von euch ist Alice und der andere ist Bob.\nAlice ist durch eine Wand von Bob getrennt. Das Ziel des Spiels ist es, eure (gemeinsame) Belohnung zu maximieren. \nAuf Alices Seite der Wand befindet sich entweder eine Katze oder ein Hund. \nZusätzlich gibt es auf Alices Seite der Wand einen Lichtschalter, den sie einschalten kann, um eine Glühbirne auf Bobs Seite der Wand zu beleuchten.\n\nAlice beginnt und kann genau eine der folgenden Aktionen ausführen:\n- aussteigen für eine (gemeinsame) Belohnung von 1 (dies beendet das Spiel sofort)\n- die Wand entfernen für eine (gemeinsame) Belohnung von -5\n- die Glühbirne einschalten (keine Belohnung)\n- nichts tun, d. h. die Glühbirne bleibt aus (keine Belohnung)\n\nWenn Alice nicht ausgestiegen ist, ist Bob an der Reihe. Bob beobachtet den Zustand des Lichts sowie – falls die Wand von Alice entfernt wurde – das Tier auf Alices Seite der Wand.\nBob kann genau eine der folgenden Aktionen ausführen:\n- aussteigen für eine (gemeinsame) Belohnung von 0.5\n- oder raten, ob das Tier eine Katze oder ein Hund ist\n    - wenn Bob richtig raten, erhält er eine (gemeinsame) Belohnung von 10\n    - wenn Bob falsch raten, erhält er eine (gemeinsame) Belohnung von -10\n\nDie Spieler können sich nicht mit dem anderen Spieler außer den oben genannten Optionen (d. h. Lichtstatus und Entfernen der Wand) kommunizieren.\n"
            },
        )

    elif env_name == "cat_and_dog_sp":
        messages.append(
            {
                "role": "user",
                "content": "\nYou are playing a cooperative game with an identical instance of this yourself. The same language model is used for both players, but each instance is assigned a different role in the game.\nThis game is generally turn-based. While you cannot communicate with the other player during the game, you can see their actions and the (joined) rewards from previous turns. Your goal is to maximize the (joint) reward.\nNote that you are only playing one round of the game (one round consists of multiple turns).\n\nNow follows a detailed description of the game and (if applicable) the history of previous turns.\n\n\nYou are playing a game of cat and dog with an identical instance of this language model. One instance is assigned the role of Alice and the other is assigned the role of Bob.\nAlice is separated from Bob by a wall. The goal of the game is maximize your (joint) reward.\nOn Alice's side of the wall, there is either a cat or a dog.\nAdditionally, there is a light switch on Alice's side of the wall, which she can turn on to illuminate a bulb on Bob's side of the wall.\n\nAlice goes first and can do exactly one of the following:\n- bail for a (joint) reward of 1 (this ends the game immediately)\n- remove the wall for a (joint) reward of -5\n- turn on the light bulb (no reward)\n- do nothing, i.e. the light bulb stays off (no reward)\n\nIf Alice didn't bail, it is Bob's turn. Bob observes the light status and the animal on Alice's side of the wall, if the wall was removed by Alice.\nBob can do exactly one of the following:\n- bail for a (joint) reward of 0.5\n- or guess the animal, i.e. either cat or dog\n    - if Bob guesses correctly, he receives a (joint) reward of 10\n    - if Bob guesses incorrectly, he receives a (joint) reward of -10\n\nThe players cannot communicate with each other except the options presented above (i.e. light status and removing the wall).\n"
            },
        )
    
    return messages

def get_chat_history_for_p_type(cfd: bool, p_type: str, animal: Optional[str] = None, light_status: Optional[str] = None, env_name: str = "cat_and_dog") -> ChatHistoryLike:
    messages = list(get_base_messages(env_name))
    if p_type == "a1_animal":
        
        if env_name == "cat_and_dog" or env_name == "cat_and_dog_sp" or env_name == "cat_and_dog_semantics":
            messages.append(
                {
                    "role": "user",
                    "content": f"\nYou are Alice. On your side of the wall, there is a {animal}.\n",
                }
            )
        elif env_name == "cat_and_dog_german":
            messages.append(
                {
                    "role": "user",
                    "content": f"\nDu bist Alice. Auf deiner Seite der Wand befindet sich eine {animal}.\n",
                }
            )
        else:
            raise ValueError(f"Unknown env_name={env_name!r}")
        
        if cfd:
            raise NotImplementedError("CFD prompts not wired up yet for this helper.")
            # messages.append(
            #     {
            #     "role": "user",
            #     "content": "\nYou will now choose an action for the current turn. You must choose one of the following legal actions:\n\nbail\nremove_wall\nturn_on_light\ndo_nothing\n\n\n\n\nEnsure that the action you take is strictly compliant with the following Coordination-friendly definition (CFD):\nAn action is NOT coordination-friendly if it satisfies ANY of the following:\n    • It assumes the existence of arbitrary conventions. To check this, ask:\n        – Is it using a signalling convention that could be replaced by a different, equivalent convention?\n        – Is there any role assignment that could be flipped? (This only applies when two players are symmetric in the game.)\n    • Note: most actions rely on some convention — you must judge whether that convention is arbitrary or essential.\n    Otherwise, the action is coordination-friendly. \n    Before suggesting an action, think through the questions above; if any apply, the action is not coordination-friendly.\n\n\nRespond in the following format:\n\nREASONING:\n<REASONING>\n\nACTION JSON:\n```json\n<JSON>\n```\n\nIn <REASONING>, provide a short explanation of your choice of action.\n\nIn <JSON>, provide the action you are going to take in this round. The JSON should have the following format:\n- \"action\": one of the legal actions. Make sure to use the exact string from the legal actions list or else the JSON will not be parsed correctly.\n\nThis JSON will be automatically parsed, so ensure the format is precise and the action matches one of the legal actions exactly.\n"
            #     }
            # )
        else:
            if env_name == "cat_and_dog" or env_name == "cat_and_dog_sp":
                messages.append(
                    {
                        "role": "user",
                        "content": "\nYou will now choose an action for the current turn. You must choose one of the following legal actions:\n\nbail\nremove_wall\nturn_on_light\ndo_nothing\n\n\n\n\nRespond in the following format:\n\nREASONING:\n<REASONING>\n\nACTION JSON:\n```json\n<JSON>\n```\n\nIn <REASONING>, provide a short explanation of your choice of action.\n\nIn <JSON>, provide the action you are going to take in this round. The JSON should have the following format:\n- \"action\": one of the legal actions. Make sure to use the exact string from the legal actions list or else the JSON will not be parsed correctly.\n\nThis JSON will be automatically parsed, so ensure the format is precise and the action matches one of the legal actions exactly.\n",
                    }
                )
            elif env_name == "cat_and_dog_semantics":
                messages.append(
                    {
                        "role": "user",
                        "content": "\nYou will now choose an action for the current turn. You must choose one of the following legal actions:\n\nbail\nremove_wall\nturn_on_red_light\nturn_on_blue_light\n\n\n\n\nRespond in the following format:\n\nREASONING:\n<REASONING>\n\nACTION JSON:\n```json\n<JSON>\n```\n\nIn <REASONING>, provide a short explanation of your choice of action.\n\nIn <JSON>, provide the action you are going to take in this round. The JSON should have the following format:\n- \"action\": one of the legal actions. Make sure to use the exact string from the legal actions list or else the JSON will not be parsed correctly.\n\nThis JSON will be automatically parsed, so ensure the format is precise and the action matches one of the legal actions exactly.\n",
                    }
                )
            elif env_name == "cat_and_dog_german":
                messages.append(
                    {
                        "role": "user",
                        "content": "\nDu wirst nun eine Aktion für den aktuellen Zug auswählen. Du musst eine der folgenden zulässigen Aktionen wählen:\n\naussteigen\nwand_entfernen\nlicht_einschalten\nnichts_tun\n\n\n\n\nAntworte im folgenden Format:\n\nREASONING:\n<REASONING>\n\nACTION JSON:\n```json\n<JSON>\n```\n\nIn <REASONING> gib eine kurze Erklärung für deine Wahl der Aktion an.\n\nIn <JSON> gib die Aktion an, die du in dieser Runde ausführen wirst. Das JSON muss folgendes Format haben:\n- \"action\": eine der zulässigen Aktionen. Stelle sicher, dass du exakt die Zeichenkette aus der Liste der zulässigen Aktionen verwendest, da das JSON sonst nicht korrekt geparst wird.\n\nDieses JSON wird automatisch geparst, stelle daher sicher, dass das Format exakt ist und die Aktion genau einer der zulässigen Aktionen entspricht.\n"
                    }
                )
            else:
                raise ValueError(f"Unknown env_name={env_name!r}")
    elif p_type == "a2_a1":
        if env_name == 'cat_and_dog' or env_name == "cat_and_dog_semantics" or env_name == "cat_and_dog_sp":
            messages.append(
                {
                    "role": "user",
                    "content": f"\nYou are Bob. The light bulb on your side of the wall is {light_status}. \nFurthermore, the wall has not been removed. You cannot see the animal on the other side of the wall."
                }
            )
        elif env_name == "cat_and_dog_german":
            messages.append(
                {
                    "role": "user",
                    "content": f"\nDu bist Bob. Die Glühbirne auf deiner Seite der Wand ist {light_status}. \nAußerdem wurde die Wand nicht entfernt. Du kannst das Tier auf der anderen Seite der Wand nicht sehen."
                }
            )
        else:
            raise ValueError(f"Unknown env_name={env_name!r}")
        if cfd:
            raise NotImplementedError("CFD prompts not wired up yet for this helper.")
        else:
            if env_name == "cat_and_dog" or env_name == "cat_and_dog_semantics" or env_name == "cat_and_dog_sp":
                messages.append(
                    {
                        "role": "user",
                        "content": "\nYou will now choose an action for the current turn. You must choose one of the following legal actions:\n\nbail\nguess_dog\nguess_cat\n\n\n\n\nRespond in the following format:\n\nREASONING:\n<REASONING>\n\nACTION JSON:\n```json\n<JSON>\n```\n\nIn <REASONING>, provide a short explanation of your choice of action.\n\nIn <JSON>, provide the action you are going to take in this round. The JSON should have the following format:\n- \"action\": one of the legal actions. Make sure to use the exact string from the legal actions list or else the JSON will not be parsed correctly.\n\nThis JSON will be automatically parsed, so ensure the format is precise and the action matches one of the legal actions exactly.\n"
                    }
                )
            elif env_name == "cat_and_dog_german":
                messages.append(
                    {
                        "role": "user",
                        "content": "\nDu wirst nun eine Aktion für den aktuellen Zug auswählen. Du musst eine der folgenden zulässigen Aktionen wählen:\n\naussteigen\nhund_raten\nkatze_raten\n\n\n\n\nAntworte im folgenden Format:\n\nREASONING:\n<REASONING>\n\nACTION JSON:\n```json\n<JSON>\n```\n\nIn <REASONING> gib eine kurze Erklärung für deine Wahl der Aktion an.\n\nIn <JSON> gib die Aktion an, die du in dieser Runde ausführen wirst. Das JSON muss folgendes Format haben:\n- \"action\": eine der zulässigen Aktionen. Stelle sicher, dass du exakt die Zeichenkette aus der Liste der zulässigen Aktionen verwendest, da das JSON sonst nicht korrekt geparst wird.\n\nDieses JSON wird automatisch geparst, stelle daher sicher, dass das Format exakt ist und die Aktion genau einer der zulässigen Aktionen entspricht.\n"
                    }
                )
            else:
                raise ValueError(f"Unknown env_name={env_name!r}")
    return messages


def filter_responses_to_context_reasoning_and_action(
    context_value: str,
    responses_all: List[List[RequestOutput]],
) -> List[List[Any]]:
    """
    Parse responses into rows:
      [context_value, {"reasoning": <str|None>, "action": <dict|None>}]

    Where `context_value` is:
      - `animal` for p_type == "a1_animal"
      - `light_status` for p_type == "a2_a1"

    Notes:
      - If parsing fails, the corresponding entry is `None`.
    """
    rows: List[List[Any]] = []

    reasoning_re = re.compile(r"REASONING:\s*(.*?)\s*ACTION JSON:", re.DOTALL | re.IGNORECASE)
    json_block_re = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

    for seed_responses in responses_all:
        for resp in seed_responses:
            text = resp.content or ""

            # Reasoning
            m = reasoning_re.search(text)
            reasoning = m.group(1).strip() if m else None
            # Action JSON
            action_dict: Optional[dict] = None
            m2 = json_block_re.search(text)
            if m2:
                json_str = m2.group(1).strip()
                try:
                    action_dict = json.loads(json_str)
                except Exception:
                    action_dict = None

            rows.append([context_value, {"reasoning": reasoning, "action": action_dict}])

    return rows


def run_condition_and_save(
    *,
    env_name: str,
    llm_type: LLMType,
    llm_name: str,
    use_CFD: bool,
    p_type: str,
    condition: str,
    n_seeds: int,
    n_queries: int,
    n_workers: int,
    animal: Optional[str] = None,
    light_status: Optional[str] = None,
    reasoning_effort: Optional[str] = "medium",
    use_langfuse: bool = True,
) -> List[List[Any]]:
    """
    One unit of data generation:
    - build chat history for (p_type, condition)
    - query the LLM many times
    - parse into [condition, {"reasoning": ..., "action": ...}] rows
    - save to ../data/action_states/<env>/<llm>/CFD_<use_CFD>/<p_type>_<condition>_responses.json
    """
    effort_suffix = ""
    if reasoning_effort in {"low", "high"}:
        effort_suffix = f"_effort_{reasoning_effort}"
    save_path = (
        f"../data/action_states/{env_name}/{llm_name}/CFD_{use_CFD}/"
        f"{p_type}_{condition}{effort_suffix}_responses.json"
    )

    messages = get_chat_history_for_p_type(
        cfd=use_CFD,
        p_type=p_type,
        animal=animal,
        light_status=light_status,
        env_name=env_name,
    )

    responses_all = query_llm_many_seeds_with_chat_history(
        messages,
        n_seeds=n_seeds,
        n_queries_per_seed=n_queries,
        llmtype=llm_type,
        reasoning_effort=reasoning_effort,
        use_langfuse=use_langfuse,
        langfuse_session_id=f"{env_name}/{llm_name}/CFD_{use_CFD}/{p_type}/{condition}/effort_{reasoning_effort}",
        max_workers=n_workers,
    )

    rows = filter_responses_to_context_reasoning_and_action(
        context_value=condition,
        responses_all=responses_all,
    )

    save_path_p = Path(save_path)
    save_path_p.parent.mkdir(parents=True, exist_ok=True)
    save_path_p.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(
        f"Saved {len(rows)} rows to {save_path_p.resolve()} "
        f"(reasoning_effort={reasoning_effort})"
    )

    return rows

if __name__ == "__main__":

    n_seeds = 1
    n_queries = 100
    n_workers = n_seeds * n_queries
    use_CFD = False
    reasoning_efforts = ["medium"]

    env_names = ["cat_and_dog_sp"]
    llm_types = [LLMType.OR_GPT_52]
    llm_names = [ "GPT-5.2"]

    for reasoning_effort in reasoning_efforts:
        for env_name in env_names:
            
            p_types = ["a1_animal", "a2_a1"]
            if env_name == "cat_and_dog" or env_name == "cat_and_dog_sp":
                light_statuses = ["on", "off"]
            elif env_name == "cat_and_dog_semantics":
                light_statuses = ["red", "blue"]
            elif env_name == "cat_and_dog_german":
                light_statuses = ["an", "aus"]
            if env_name == "cat_and_dog" or env_name == "cat_and_dog_semantics" or env_name == "cat_and_dog_sp":
                animals = ["cat", "dog"]
            elif env_name == "cat_and_dog_german":
                animals = ["katze", "hund"]
            else:
                raise ValueError(f"Unknown env_name={env_name!r}")

            for llm_type in llm_types:
                llm_name = llm_names[llm_types.index(llm_type)]

                for p_type in p_types:
                    if p_type == "a1_animal":
                        for animal in animals:
                            condition = animal
                            run_condition_and_save(
                                env_name=env_name,
                                llm_type=llm_type,
                                llm_name=llm_name,
                                use_CFD=use_CFD,
                                p_type=p_type,
                                condition=condition,
                                n_seeds=n_seeds,
                                n_queries=n_queries,
                                n_workers=n_workers,
                                animal=animal,
                                light_status=None,
                                reasoning_effort=reasoning_effort,
                            )

                    elif p_type == "a2_a1":
                        for light_status in light_statuses:
                            condition = f"light_{light_status}"
                            run_condition_and_save(
                                env_name=env_name,
                                llm_type=llm_type,
                                llm_name=llm_name,
                                use_CFD=use_CFD,
                                p_type=p_type,
                                condition=condition,
                                n_seeds=n_seeds,
                                n_queries=n_queries,
                                n_workers=n_workers,
                                animal=None,
                                light_status=light_status,
                                reasoning_effort=reasoning_effort,
                            )
                    else:
                        raise ValueError(f"Unknown p_type={p_type!r}")
    