import asyncio
import pytest
from unittest.mock import patch
from rubric_rl.refinement import single_pass_worker as refinement
from rubric_rl.refinement.protocol import PositiveRefinementProtocolError


def _parent():
    return {
        "rubric_id": "parent-1",
        "parent_fingerprint": "fp-1",
        "title": "Parent",
        "description": "Parent requirement",
        "weight": 1.0,
    }


def test_five_candidates_fill_the_fixed_pool():
    parent = _parent()
    job = {"blend_lambda": 0.5}

    async def generator_call(prompt, schema, *, config):
        assert schema["properties"]["candidates"]["minItems"] == 5
        assert schema["properties"]["candidates"]["maxItems"] == 5
        return (
            {
                "candidates": [
                    {"title": f"Child {i}", "completion_point": f"Point {i}"}
                    for i in range(5)
                ]
            },
            {"prompt_character_count": 1},
        )

    with (
        patch.object(
            refinement.refinement_worker, "_validate_seed_candidates", return_value=[]
        ),
        patch.object(
            refinement.refinement_worker, "_generator_prompt", return_value="prompt"
        ),
        patch.object(
            refinement.refinement_worker, "_reindex_candidates", side_effect=lambda x: x
        ),
        patch.object(
            refinement.refinement_worker,
            "_fixed_meta_valid_children",
            side_effect=lambda x, max_children: (x[:max_children], []),
        ),
    ):
        pool, _audit = asyncio.run(
            refinement._prepare_parent(
                job, parent, object(), generator_call=generator_call
            )
        )
    assert len(pool) == 5


def test_three_candidates_are_rejected_as_below_effective_minimum():
    with pytest.raises(PositiveRefinementProtocolError, match="expected=5..5"):
        refinement.refinement_worker._validate_generated(
            {
                "candidates": [
                    {"title": f"Child {i}", "completion_point": f"Point {i}"}
                    for i in range(3)
                ]
            },
            min_candidate_count=5,
            max_candidate_count=5,
            parent=_parent(),
            polarity="positive",
        )
