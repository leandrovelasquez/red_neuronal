from pathlib import Path
def read_class_names(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
