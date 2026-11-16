from data.dataloader import clean_data, load_data


def test_clean_data_removes_headers_urls_and_normalizes_whitespace():
    raw = "Subject: Hello World\nFrom: person@example.com\nVisit https://example.com\nUseful   TEXT!"
    cleaned = clean_data(raw)

    assert "person@example.com" not in cleaned
    assert "https" not in cleaned
    assert "useful text" in cleaned


def test_load_data_reads_category_folders(tmp_path):
    category = tmp_path / "comp.graphics"
    category.mkdir()
    (category / "001").write_text(
        "Subject: Rendering\n3D graphics are useful.", encoding="utf-8"
    )

    documents, labels = load_data(tmp_path)

    assert documents == ["3d graphics are useful"]
    assert labels == ["comp.graphics"]
