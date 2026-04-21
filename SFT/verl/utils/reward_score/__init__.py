# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated

# 去除prompt_str


def default_compute_score(
    data_source,
    prompt_str,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    val_type='f1',
    batch_size=1,
    is_valid=True,
    tokenizer = None
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if type(data_source) != str:
        reslist = []
        if val_type == 'llm':
            # llm只支持batch模式
            from . import llm_judge
            reslist = llm_judge.compute_score_batch(prompt_str, solution_str, ground_truth, data_source, batch_size, is_valid, tokenizer)
        else:
            for data_source_e, solution_str_e, ground_truth_e in zip(data_source, solution_str, ground_truth):
                if data_source_e in ['nq', "2wiki", "Bamboogle", "hotpotqa", "musique", "tq", "popqa", "browse_comp", "browse_comp_zh", "xbench_deepsearch", "hotpot", "zhihu", "webshaper"] or 'browse_comp' in data_source_e:
                    from . import format_and_f1
                    res = format_and_f1.compute_score(solution_str_e, ground_truth_e, data_source_e, val_type=val_type)
                    reslist.append(res)
                elif data_source_e in ['future']:
                    from . import stock_judge
                    res = stock_judge.compute_score(solution_str_e, ground_truth_e, data_source_e, val_type=val_type)
                    reslist.append(res)
                elif data_source_e in ['Factbench', 'politifact', 'liar2', 'elobench', 'Chinese_Rumor_Dataset', 'fever', 'lair', 'MDFEND-Weibo21', 'twitter_factchecking_test', 'factchecker_history', 'elobench', 'health_fact'] or 'browse_comp' in data_source_e:
                    from . import fact_test
                    res = fact_test.compute_score(solution_str_e, ground_truth_e, data_source_e, val_type=val_type)
                    reslist.append(res)
                else:
                    raise NotImplementedError(f"Reward function is not implemented for {data_source=}")
        return reslist
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward

        res = math_reward.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ['nq', "2wiki", "Bamboogle", "hotpotqa", "musique", "tq", "popqa", "browse_comp", "browse_comp_zh", "xbench_deepsearch", "hotpot", "zhihu", "webshaper"]:
        from . import format_and_f1
        res = format_and_f1.compute_score(solution_str, ground_truth, data_source, val_type=val_type)
    elif data_source in ['Factbench', 'politifact', 'liar2', 'elobench', 'Chinese_Rumor_Dataset', 'fever', 'lair', 'MDFEND-Weibo21', 'twitter_factchecking_test', 'factchecker_history', 'elobench', 'health_fact']:
        from . import fact_test
        res = fact_test.compute_score(solution_str, ground_truth, data_source, val_type=val_type)

    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    prompt_str,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, prompt_str, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb

    )


__all__ = ["default_compute_score"]
