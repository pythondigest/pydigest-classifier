import json

import pytest

from src.api.server import app  # Импортируем ваше Flask приложение


# Настраиваем тестовый клиент
@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# Фиктивный классификатор для тестирования
class DummyClassifier:
    def predict(self, data):
        # Возвращает фиктивные данные для предсказания
        return [True if "link" in item else False for item in data]


# Замена классификатора на DummyClassifier
def test_classify(client, monkeypatch):
    # Создаём экземпляр классификатора и замещаем оригинальный
    dummy_classifier = DummyClassifier()
    monkeypatch.setattr("src.api.server.classifier", dummy_classifier)

    # Подготовка тестовых данных
    test_data = {
        "links": [
            {
                "link": "http://www.python.org/dev/peps/pep-0453/",
                "data": {
                    "title": "Pip будет включен в поставку Python 3.4",
                    "description": "Это решение из PEP одобрено 22 октября 2013 г.",
                },
            }
        ]
    }

    # Проведение POST-запроса к нашему API
    response = client.post("/api/v1.0/classify/", json=test_data)

    # Проверка ответа
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert "links" in response_json
    assert len(response_json["links"]) == len(test_data["links"])
    assert response_json["links"][0]["http://www.python.org/dev/peps/pep-0453/"] is True


def test_status(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert response.data.decode() == "OK"


def test_redirect(client):
    response = client.get("/")
    assert response.status_code == 302
    assert response.location == "https://pythondigest.ru/"


# Тест на ошибки, когда поле "links" отсутствует
def test_classify_no_links(client):
    response = client.post("/api/v1.0/classify/", json={})
    assert response.status_code == 400
    assert "links" in response.get_json()["error"]


# Тест на ошибки, когда ссылки неверного формата
def test_classify_invalid_link_format(client):
    test_data = {"links": [{"link": ""}]}
    response = client.post("/api/v1.0/classify/", json=test_data)
    assert response.status_code == 400
    assert "invalid link format" in response.get_json()["error"]
