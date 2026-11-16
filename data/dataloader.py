from pathlib import Path
import re

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "raw" / "20_newsgroups"


def clean_data(text: str) -> str:
    """Normalize a newsgroup post and remove common headers and URLs."""
    text = re.sub(r"^[\w-]+: .+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def load_data(data_dir: str | Path = DEFAULT_DATA_DIR) -> tuple[list[str], list[str]]:
    """Load category-folder documents into parallel document and label lists."""
    root = Path(data_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    documents: list[str] = []
    labels: list[str] = []
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for file_path in sorted(p for p in category_dir.iterdir() if p.is_file()):
            cleaned = clean_data(file_path.read_text(encoding="latin-1"))
            if cleaned:
                documents.append(cleaned)
                labels.append(category_dir.name)

    if not documents:
        raise ValueError(f"No documents found in dataset directory: {root}")
    return documents, labels
