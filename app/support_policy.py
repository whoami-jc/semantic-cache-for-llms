"""Conservative English FAQ scope for the store-support demo (not a PII detector)."""
import re

# Small recall tradeoff measured on existing queries; see validation/cutoff-comparison.
RETRIEVAL_THRESHOLD = 0.78


class SupportFAQPolicy:
    revision = "store-faq-v1"
    topics = re.compile(
        r"\b(return\w*|refund\w*|ship\w*|delivery|deliver\w*|payment\w*|pay|"
        r"checkout|exchange\w*|cancel\w*)\b", re.I,
    )
    personalized = re.compile(
        r"[\d@#$€£]|\b(my|mine|our|ours|account|tracking|track|status|"
        r"today|yesterday|tomorrow|currently|now|ordered|charged|"
        r"one|two|three|four|five|six|seven|eight|nine|ten|hundred|"
        r"dollars?|euros?|pounds?)\b", re.I,
    )

    def allows(self, prompt: str) -> bool:
        # Bypass both reads and writes, including exact hits, for excluded requests.
        return bool(self.topics.search(prompt)) and not self.personalized.search(prompt) and len(prompt) <= 300
