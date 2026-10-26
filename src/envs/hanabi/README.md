# Hanabi

`hanabi.py` and `hanabi_game.py` are adapted from the JAX Hanabi environment in
[JaxMARL](https://github.com/FLAIROx/JaxMARL) (Apache License 2.0; see the repository's
`LICENSE` and `NOTICE`), with minor bug fixes. `hanabi_text_env.py` wraps it as a text
environment for LLM agents; the paper uses the 3x3 variant (three colours, three ranks,
one life token and three hint tokens).
