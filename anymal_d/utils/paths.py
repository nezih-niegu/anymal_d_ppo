import os


def find_project_file(rel_path: str) -> str:
    """Locate *rel_path* by walking up from this file and the CWD.

    Works regardless of which directory the user launches scripts from.
    """
    seeds = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    for seed in seeds:
        d = seed
        for _ in range(8):
            cand = os.path.join(d, rel_path)
            if os.path.exists(cand):
                return cand
            d = os.path.dirname(d)
    return os.path.join(".", rel_path)


MODEL_XML: str = find_project_file(os.path.join("anybotics_anymal_d", "scene.xml"))
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(MODEL_XML), ".."))
SAVE_DIR: str = os.path.join(PROJECT_ROOT, "pretrained_models", "anymal_d")
VIDEO_DIR: str = os.path.join(SAVE_DIR, "videos")
