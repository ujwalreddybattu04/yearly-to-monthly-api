import pytest
from fastapi.testclient import TestClient
from main import app, _read_csv

@pytest.mark.parametrize("ending", ["\r", "\n", "\r\n"])
@pytest.mark.parametrize("mode", ["average", "exit"])
def test_csv_line_endings(ending, mode):
    content = ("\ufeff" + ending.join(["year,value", "1,6%", "2,8%", "3,12%", ""]) + "\r\n").encode("utf-8")
    response = TestClient(app).post("/convert", params={"mode":mode,"format":"json"}, files={"file":("a.csv",content)})
    assert response.status_code == 200, response.text
    assert len(response.json()) == 36


def test_quoted_newlines_preserved():
    frame = _read_csv(b'year,"multi\rline"\r1,6%\r2,8%\r\n')
    assert list(frame.columns) == ["year", "multi\rline"]
    assert len(frame) == 2
