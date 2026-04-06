import random

MAX_SEED = 0x7FFFFFFFFFFFFFFF  # 2^63 - 1


class JLC_SeedGenerator:
    def __init__(self):
        self.last_seed = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_SEED,
                    },
                ),
                "spacer": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True
                    }
                ),
            },
        }

    RETURN_TYPES = ("SEED", "INT")
    RETURN_NAMES = ("seed", "seed_int")
    OUTPUT_NODE = False
    FUNCTION = "generator"
    CATEGORY = "seed"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return None

    def generator(self, seed=0, spacer=None):
        # ✅ ensure valid seed on first load
        if seed is None:
            seed = 0

        # 🔒 Clamp for safety
        current_seed = max(0, min(int(seed), MAX_SEED))

        # 🧠 Store last (future use)
        self.last_seed = current_seed

        return {
            "result": ({"seed": current_seed}, current_seed),
            "ui": {"seed": str(current_seed)}
        }