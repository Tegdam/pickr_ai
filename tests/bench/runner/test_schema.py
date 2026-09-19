import pytest

from bench.runner.schema import REQUIRED_CLIENT_KEYS, SchemaError, validate_client_output


def test_valid_client_json_passes(client_json):
    validate_client_output(client_json)  # must not raise


def test_missing_key_raises_naming_it(client_json):
    del client_json["ttfts"]
    with pytest.raises(SchemaError, match="ttfts"):
        validate_client_output(client_json)


def test_wrong_type_raises_naming_the_key(client_json):
    client_json["completed"] = "4"
    with pytest.raises(SchemaError, match="completed"):
        validate_client_output(client_json)


def test_bool_is_not_accepted_as_int_or_float(client_json):
    """bool is a subclass of int in Python -- must not slip past the numeric check."""
    client_json["duration"] = True
    with pytest.raises(SchemaError, match="duration"):
        validate_client_output(client_json)


def test_length_mismatch_raises_mentioning_length(client_json):
    client_json["ttfts"] = client_json["ttfts"][:-1]
    with pytest.raises(SchemaError, match="length"):
        validate_client_output(client_json)


def test_required_keys_cover_the_doc_minimum():
    """doc §5/§8: at minimum these keys with these types."""
    expected = {
        "completed": int, "duration": float, "total_input_tokens": int, "total_output_tokens": int,
        "request_throughput": float, "output_throughput": float, "ttfts": list, "itls": list,
        "input_lens": list, "output_lens": list, "generated_texts": list, "errors": list, "start_times": list,
    }
    for key, typ in expected.items():
        assert REQUIRED_CLIENT_KEYS[key] is typ
