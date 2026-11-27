HISTORICAL = (("O homem", "Na cidade"), ("O homem", "A mulher"), ("A manhã", "A noite"))
MATCHED = (
    ("O menino", "O médico", " estava com"),
    ("Na floresta", "Na montanha", " havia um"),
    ("O soldado", "Um pintor", " pegou a"),
    ("Ela", "Ele", " estava muito"),
)
PREFIXES = (("O menino", "O médico"), ("O soldado", "Um pintor"), ("Ela", "Ele"))
SUFFIXES = {
    8: " passou ",
    16: " chegou em casa ",
    32: " entrou na sala e ficou em pé ",
    64: " voltou para casa, abriu a porta da sala e encontrou sobre a mesa ",
    96: " acordou cedo, abriu a janela e viu que a chuva tinha parado. Depois saiu de casa para buscar ",
}


def study_prompts(vocabulary, context):
    def pair(left, right, suffix):
        assert len(left) == len(right)
        return {"prompts": [left + suffix, right + suffix], "suffix": suffix, "suffix_length": len(suffix)}

    study = {
        "historical": [{"prompts": list(pair)} for pair in HISTORICAL],
        "matched": [pair(*entry) for entry in MATCHED],
        "distance": [
            {"target_length": target, **pair(left, right, suffix)}
            for target, suffix in SUFFIXES.items()
            for left, right in PREFIXES
        ],
    }
    for entries in study.values():
        for entry in entries:
            for prompt in entry["prompts"]:
                assert len(prompt) <= context and set(prompt) <= vocabulary.keys()
    return study
