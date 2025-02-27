from collections import Counter
from typing import List


class RewardMethods:
    @staticmethod
    def majority_element(numbers: List[str]) -> str | None:
        if not numbers:
            return None

        count = Counter(numbers)
        majority_count = len(numbers) // 2

        for num, cnt in count.items():
            if cnt > majority_count:
                return num

        return None
