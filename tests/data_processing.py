def test_load_data():
    from src.data_processing import load_data
    df = load_data("data/raw/data.csv")
    assert not df.empty
    assert "Amount" in df.columns
