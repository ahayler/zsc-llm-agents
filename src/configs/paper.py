"""Published experiment settings; kept separate from training success criteria."""

CORE_TEXT_EPISODES = {
    "cat_and_dog": 100,
    "ad_hoc_conventions": 50,
    "ad_hoc_conventions_permuted_signals": 50,
    "tiny_hanabi": 100,
    "tiny_hanabi_fixed": 100,
    "brittle_bridge": 50,
    "narrow_tunnel": 50,
    "permuted_levers": 50,
    "matrix_game": 100,
}
CORE_TEXT_MODELS = ("OR_GPT_52", "OR_GROK_41_FAST")
CORE_CODE_MODELS = (
    "OR_GROK_41_FAST", "OR_GPT_52", "OR_GEMINI_PRO_3_PREVIEW",
    "OR_CLAUDE_OPUS_45", "OR_DEEPSEEK_V3_2",
)
CFD_TRANSFER_MODELS = CORE_CODE_MODELS[:-1]
CAT_DOG_XP_MODELS = (
    "OR_GPT_52", "OR_GROK_41_FAST", "OR_GEMINI_PRO_3_PREVIEW", "OR_DEEPSEEK_V3_2",
)
HANABI_HINT_GUESS_MODELS = (
    "OR_GROK_41_FAST", "OR_GEMINI_PRO_31_PREVIEW", "OR_GPT54", "OR_CLAUDE_SONNET_46",
)
CODE_XP_MATRICES = 6
CODE_EPISODES = 100
TEXT_XP_EPISODES = 10


def handcrafted_cfd(subject: str) -> str:
    article = "an" if subject == "action" else "a"
    plural = "actions" if subject == "action" else "policies"
    return f"""{article.capitalize()} {subject} is NOT coordination-friendly if it satisfies ANY of the following:
• It assumes the existence of arbitrary conventions. To check this, ask:
    – Is it using a signalling convention that could be replaced by a different, equivalent convention?
    – Is there any role assignment that could be flipped? (This only applies when two players are symmetric in the game.)
• Note: most {plural} rely on some convention — you must judge whether that convention is arbitrary or essential.
Otherwise, the {subject} is coordination-friendly.
Before suggesting {article} {subject}, think through the questions above; if any apply, the {subject} is not coordination-friendly.
"""


LEARNED_CFD = (
    "Prefer actions that would still be reasonable under several plausible partner "
    "interpretations of the shared observation, and avoid actions that are good only "
    "if a partner infers one specific unstated rationale."
)
TEXT_CFDS = {"baseline": "", "handcrafted": handcrafted_cfd("action"), "learned": LEARNED_CFD}
CODE_CFDS = {"baseline": "", "handcrafted": handcrafted_cfd("policy"), "learned": LEARNED_CFD}

# Failure means episode return AT OR BELOW the threshold (as in the paper's threshold table).
# Ad-hoc Conventions returns move in steps of 0.5, so <= 49.5 means below the safe reveal return of 50.
# These are not the sufficient_score values used to select training policies.
FAILURE_THRESHOLDS = {
    "cat_and_dog": 0.0,
    "permuted_levers": 0.0,
    "brittle_bridge": -20.0,
    "narrow_tunnel": -20.0,
    "ad_hoc_conventions": 49.5,
    "ad_hoc_conventions_permuted_signals": 49.5,
    "tiny_hanabi": 0.0,
    "tiny_hanabi_fixed": 0.0,
    "matrix_game": 0.0,
}
