# TokenTracker moved from main.py to avoid circular import

class TokenTracker:
    def __init__(self, display):
        self.total_cache_creation = 0
        self.total_cache_retrieval = 0
        self.total_input = 0
        self.total_output = 0
        self.recent_cache_creation = 0
        self.recent_cache_retrieval = 0
        self.recent_input = 0
        self.recent_output = 0
        self.displayA = display

    def update(self, response):
        self.recent_cache_creation = response.usage.cache_creation_input_tokens
        self.recent_cache_retrieval = response.usage.cache_read_input_tokens
        self.recent_input = response.usage.input_tokens
        self.recent_output = response.usage.output_tokens
        self.total_cache_creation += self.recent_cache_creation
        self.total_cache_retrieval += self.recent_cache_retrieval
        self.total_input += self.recent_input
        self.total_output += self.recent_output

    def display(self, displayA):
        recent_usage = [
            "Recent Token Usage 📊",
            f"Recent Cache Creation: {self.recent_cache_creation:,}",
            f"Recent Cache Retrieval: {self.recent_cache_retrieval:,}",
            f"Recent Input: {self.recent_input:,}",
            f"Recent Output: {self.recent_output:,}",
            f"Recent Total: {self.recent_cache_creation + self.recent_cache_retrieval + self.recent_input + self.recent_output:,}",
        ]
        total_cost = (
            self.total_cache_creation * 3.75 +
            self.total_cache_retrieval * 0.30 +
            self.total_input * 3 +
            self.total_output * 15
        ) / 1_000_000
        total_usage = [
            "Total Token Usage 📈",
            f"Total Cache Creation: {self.total_cache_creation:,}",
            f"Total Cache Retrieval: {self.total_cache_retrieval:,}",
            f"Total Output: {self.total_output:,}",
            f"Total Tokens: {self.total_cache_creation + self.total_cache_retrieval + self.total_input + self.total_output:,} with a total cost of ${total_cost:.2f} USD.",
        ]
        token_display = f"\n{total_usage}"
        self.displayA.add_message("user", token_display)
