import json
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from study.loader import load

ex = load(__file__, "ex02_api_serialization")

JOB_ID = "6f1c2b9e-8a7d-4c1e-9f3a-2b5d7e8f9a01"
API_PAYLOAD = {
    "jobId": JOB_ID,
    "modelName": "text-embedding-3-small",
    "inputTexts": ["hello", "world"],
    "createdAt": "2024-05-01T12:00:00Z",
    "dimensions": 256,
}


@pytest.mark.topic("py.pydantic.model_validate", "py.pydantic.api_validation")
def test_from_api_dict_camel_case():
    job = ex.EmbeddingJob.from_api(API_PAYLOAD)
    assert job.job_id == UUID(JOB_ID)
    assert job.model_name == "text-embedding-3-small"
    assert job.created_at == datetime(2024, 5, 1, 12, tzinfo=UTC)
    assert job.user_tag is None


@pytest.mark.topic("py.pydantic.model_validate", "py.pydantic.serialization")
@pytest.mark.parametrize("encode", [str, bytes])
def test_from_api_raw_json(encode):
    raw = json.dumps(API_PAYLOAD)
    job = ex.EmbeddingJob.from_api(raw if encode is str else raw.encode())
    assert job.dimensions == 256
    assert job.input_texts == ["hello", "world"]


@pytest.mark.topic("py.pydantic.model_validate")
def test_from_api_accepts_snake_case_too():
    job = ex.EmbeddingJob.from_api(
        {"job_id": JOB_ID, "model_name": "m", "input_texts": [], "created_at": "2024-01-01T00:00:00"}
    )
    assert job.model_name == "m"


@pytest.mark.topic("py.pydantic.model_validate", "py.pydantic.api_validation")
@pytest.mark.parametrize(
    "bad",
    [
        {**API_PAYLOAD, "jobId": "not-a-uuid"},
        {k: v for k, v in API_PAYLOAD.items() if k != "modelName"},
        '{"jobId": "' + JOB_ID + '", ',  # truncated JSON body
    ],
)
def test_from_api_invalid(bad):
    with pytest.raises(ValidationError):
        ex.EmbeddingJob.from_api(bad)


@pytest.mark.topic("py.pydantic.model_dump", "py.pydantic.serialization")
def test_to_api_shape():
    job = ex.EmbeddingJob.from_api({**API_PAYLOAD, "apiKey": "sk-secret"})
    body = job.to_api()
    assert body == {
        "jobId": JOB_ID,
        "modelName": "text-embedding-3-small",
        "inputTexts": ["hello", "world"],
        "createdAt": "2024-05-01T12:00:00Z",
        "dimensions": 256,
        "numInputs": 2,
    }
    assert "sk-secret" not in json.dumps(body)
    assert job.api_key.get_secret_value() == "sk-secret"


@pytest.mark.topic("py.pydantic.serialization")
def test_created_at_converted_to_utc_z():
    ist = timezone(timedelta(hours=5, minutes=30))
    job = ex.EmbeddingJob(
        job_id=UUID(JOB_ID),
        model_name="m",
        input_texts=["a"],
        created_at=datetime(2024, 5, 1, 17, 30, 15, 999, tzinfo=ist),
    )
    assert job.to_api()["createdAt"] == "2024-05-01T12:00:15Z"
    naive = job.model_copy(update={"created_at": datetime(2024, 5, 1, 9, 0)})
    assert naive.to_api()["createdAt"] == "2024-05-01T09:00:00Z"


@pytest.mark.topic("py.pydantic.model_dump")
def test_to_db_row_python_mode():
    row = ex.EmbeddingJob.from_api(API_PAYLOAD).to_db_row()
    assert row["job_id"] == UUID(JOB_ID)
    assert isinstance(row["created_at"], datetime)
    assert row["user_tag"] is None  # None kept
    assert row["num_inputs"] == 2
    assert "api_key" not in row


@pytest.mark.topic("py.pydantic.serialization", "py.pydantic.model_validate")
def test_round_trip():
    job = ex.EmbeddingJob.from_api(API_PAYLOAD)
    again = ex.EmbeddingJob.from_api(json.dumps(job.to_api()))
    assert again == job


@pytest.mark.topic("py.pydantic.model_dump", "py.pydantic.api_validation")
def test_patch_body_only_sends_explicitly_set_fields():
    assert ex.patch_body(ex.JobUpdate()) == {}
    assert ex.patch_body(ex.JobUpdate(user_tag="eval-run")) == {"userTag": "eval-run"}
    # explicit None means "clear it" and MUST be sent
    assert ex.patch_body(ex.JobUpdate(user_tag=None, dimensions=512)) == {
        "userTag": None,
        "dimensions": 512,
    }
    from_wire = ex.JobUpdate.model_validate({"modelName": "v2"})
    assert ex.patch_body(from_wire) == {"modelName": "v2"}
