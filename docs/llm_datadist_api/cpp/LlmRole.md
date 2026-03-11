# LlmRole

LLM-DataDist的角色

```
enum class LlmRole : int32_t {
  kPrompt = 1,      // 角色为Prompt
  kDecoder = 2,     // 角色为Decoder
  kMix = 3,         // 角色为Mix
  kEnd              // 无效值
}
```
