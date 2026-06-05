from src.services.id_generator import generate_id


def test_generate_id_prefixes():
    assert generate_id("resp").startswith("resp_")
    assert generate_id("msg").startswith("msg_")
    assert generate_id("call").startswith("call_")
