import pytest
from jsonschema import Draft202012Validator

from iwi_harness import EditingSession
from iwi_harness.targets import tiny_target


def write_validator(session):
    declaration = next(tool["function"] for tool in session.tools if tool["function"]["name"] == "write_weights")
    Draft202012Validator.check_schema(declaration["parameters"])
    return Draft202012Validator(declaration["parameters"])


def test_write_schema_accepts_all_operation_forms(tmp_path):
    session = EditingSession(tiny_target(), artifact_dir=tmp_path, detailed_writes=True)
    validator = write_validator(session)
    for operation, selection, operands in (
        ("copy", [5], {"source": {"tensor": "embed.weight", "selection": [3]}}),
        ("replace", [5], {"value": {"op": "scale", "left": {"tensor": "embed.weight", "selection": [3]}, "factor": 1}}),
        ("add", [5, 0], {"value": 0}),
        ("fill", [5], {"value": 0}),
        ("scale", [5], {"factor": 1}),
        ("low_rank_add", [[0, 1], [0, 1]], {"left": {"values": [[0]]}, "right": {"values": [[1]]}}),
        ("low_rank_add", [[0, 1], [0, 1]], {"left": {"values": [[0]]}, "right": {"values": [[1]]}, "factor": 0.5}),
    ):
        args = dict(tensor="embed.weight", operation=operation, selection=selection, **operands)
        validator.validate(args)
        session.call("write_weights", args)


def test_rejected_inputs_explain_operands_and_shapes_and_preserve_handles(tmp_path):
    session = EditingSession(tiny_target(), artifact_dir=tmp_path, detailed_writes=True)
    validator = write_validator(session)
    row = session.call("read_weights", {"tensor": "embed.weight", "selection": [3]})
    invalid = (
        ("replace", {"value": 3, "factor": 1}, "remove factor"),
        ("copy", {"value": 3}, "supply source and remove value"),
        ("add", {"value": {"value": 3}}, "for value"),
        ("fill", {"value": {"values": [1]}}, "numeric scalar"),
        ("low_rank_add", {"left": {"values": [[1]]}}, "supply right"),
    )
    for operation, operands, error in invalid:
        args = dict(tensor="embed.weight", selection=[5], operation=operation, **operands)
        assert not validator.is_valid(args)
        with pytest.raises(ValueError, match=error):
            session.call("write_weights", args)
    # Whether a scalar can replace the selection depends on the target shape.
    args = dict(tensor="embed.weight", selection=[5], operation="replace", value=3)
    validator.validate(args)
    with pytest.raises(ValueError, match=r"replace expects value shape \[8\].*received \[\]") as rejected:
        session.call("write_weights", args)
    assert "row selection" in str(rejected.value)
    assert "fill" not in str(rejected.value)
    assert session.store.revision == 0
    request = {"tensor": "embed.weight", "selection": [5], "operation": "copy", "source": {"handle": row["handle"]}}
    validator.validate(request)
    assert session.call("write_weights", request)["changed_parameters"] == 2



def test_compact_presentation_preserves_worker_validation(tmp_path):
    session = EditingSession(tiny_target(), artifact_dir=tmp_path)
    wrong = {"tensor": "embed.weight", "selection": [5], "operation": "replace", "value": 3, "factor": 1}
    assert write_validator(session).is_valid(wrong)
    with pytest.raises(ValueError, match="remove factor"):
        session.call("write_weights", wrong)
    assert session.store.revision == 0
