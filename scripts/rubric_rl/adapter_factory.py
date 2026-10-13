from rubric_rl.refinement import PositiveRubricRefinementAdapter


def build_rubric_adaptation_adapter(*, positive_refinement_config, run_id):
    return PositiveRubricRefinementAdapter.from_mapping(
        positive_refinement_config, run_id=run_id
    )
