
# Model constants
kimi2 = "moonshotai/kimi-k2"
openai41mini = "openai/gpt-4.1-mini"
gemma3n4b = "google/gemma-3n-e4b-it"
sonnet4 = "anthropic/claude-sonnet-4"
openai41 = "openai/gpt-4.1"
openaio3 = "openai/o3"
openaio3pro = "openai/o3-pro"
googlepro = "google/gemini-2.5-pro-preview"
googleflash = "google/gemini-2.5-flash-preview-09-2025"
googleflashlite = "google/gemini-2.5-flash-lite-preview-09-2025"
# googleflashlite duplicate removed
grok4 = "x-ai/grok-4"
grok4fast = "x-ai/grok-4-fast" # Kept this one, removed the duplicate re-definition at the end
qwen3 = "qwen/qwen3-235b-a22b-07-25"
qwencoder = "qwen/qwen3-coder"
zai45 = "z-ai/glm-4.5"
halfa = "openrouter/horizon-beta"
autor = "deepseek/deepseek-r1-0528:free"
openaiopen = "openai/gpt-oss-120b"
open5 = "openai/gpt-5"
open5mini="openai/gpt-5-mini"
open5nano = "openai/gpt-5-nano"
grokfast = "x-ai/grok-code-fast-1"
codex = "openai/gpt-5-codex"
sonnet45 = "anthropic/claude-sonnet-4.5"
ernie = "baidu/ernie-4.5-21b-a3b-thinking"
oss20 = "openai/gpt-oss-20b"
glm46 = "z-ai/glm-4.6"
haiku45="anthropic/claude-haiku-4.5"
minimax = "minimax/minimax-m2"
katcode= "kwaipilot/kat-coder-pro:free"
liquid = "deepseek/deepseek-v3.1-terminus"
codex51 = "openai/gpt-5.1-codex"
sherthink = "openrouter/sherlock-think-alpha"
sherdash = "openrouter/sherlock-dash-alpha"
gemini3pro = "google/gemini-3-pro-preview"

# Re-definition from original file (line 109 was grok4fast = "x-ai/grok-4.1-fast")
# The original file had grok4fast defined twice. 
# Line 84: grok4fast = "x-ai/grok-4-fast"
# Line 109: grok4fast = "x-ai/grok-4.1-fast"
# The second one overwrites the first. I will use the second one as it was effectively the active one.
grok4fast = "x-ai/grok-4.1-fast"

ALL_MODELS = grok4fast    # Default model for all tasks

SUMMARY_MODEL = ALL_MODELS  #   QModel for summaries
MAIN_MODEL = ALL_MODELS  # Primary model for main agent operations
CODE_MODEL = ALL_MODELS  # Model for code generation tasks
CODE_LIST = [grok4fast]  # List of models suitable for code generation
