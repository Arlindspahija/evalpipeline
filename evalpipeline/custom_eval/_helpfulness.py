import os
from typing import Tuple
import math
import re

from typing_extensions import override

from promptflow.client import load_flow

class HelpfulnessEvaluator():

    _PROMPTY_FILE = "_helpfulness.prompty"
    _RESULT_KEY = "helpfulness"

    def parse_quality_evaluator_reason_score(self, llm_output: str) -> Tuple[float, str]:
        """Parse the output of prompt-based quality evaluators that return a score and reason.

        Current supported evaluators:
            - Fluency
            - Relevance
            - Retrieval
            - Groundedness
            - Coherence

        :param llm_output: The output of the prompt-based quality evaluator.
        :type llm_output: str
        :return: The score and reason.
        :rtype: Tuple[float, str]
        """
        score = math.nan
        reason = ""
        if llm_output:
            score_pattern = r"<S2>(.*?)</S2>"
            reason_pattern = r"<S1>(.*?)</S1>"
            score_match = re.findall(score_pattern, llm_output, re.DOTALL)
            reason_match = re.findall(reason_pattern, llm_output, re.DOTALL)
            if score_match:
                score = float(score_match[0].strip())
            if reason_match:
                reason = reason_match[0].strip()

        return score, reason

    @override
    def __init__(self, model_config):
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)    
        self._flow = load_flow(source=prompty_path, model={"configuration": model_config})

    @override
    def __call__(
        self,
        *,
        query,
        context,
        response,
        **kwargs,
    ):

        llm_response = self._flow(query=query, context=context, response=response)
        score, reason = self.parse_quality_evaluator_reason_score(llm_response)

        return {
            self._RESULT_KEY: float(score),
            f"{self._RESULT_KEY}_reason": reason,
        }