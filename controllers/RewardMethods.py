from collections import Counter
from typing import List


class RewardMethods:
    @staticmethod
    def majority_element(answer_options: List[str]) -> str | None:
        """
        Find the majority element (the most repeated answer) in the list of answers.
        Args:
            answer_options: List of answer options.

        Returns:
            Majority element if it exists, otherwise None. None is also returned if all the answers are unique.
        """
        if not answer_options:
            return None

        count = Counter(answer_options)
        majority_count = len(answer_options) // 2

        for num, cnt in count.items():
            if cnt > majority_count:
                return num

        return None

    # TODO ranking
    # TODO reward model
