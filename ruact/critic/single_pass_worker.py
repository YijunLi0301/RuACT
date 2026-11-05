from .feedback_worker import CANDIDATE_BUDGET, _prepare_parent, process_feedback_job

EXECUTION_MODE = "single_pass_batched"
MIN_EFFECTIVE_CANDIDATES = CANDIDATE_BUDGET


async def process_single_pass_job(job, *, config, parent_workers):
    return await process_feedback_job(
        job, config=config, parent_workers=parent_workers, execution_mode=EXECUTION_MODE
    )
