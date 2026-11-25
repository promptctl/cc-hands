// The model the brain runs on and the context its session has used, as Claude Code itself counts them, posted to hands as
// the session starts. `tokens` is absent where Claude Code has counted none, and JSON leaves it out (docs/claude-code-behavior.md).
export function register(on) {
  on("session.start", async ($, e, next) => {
    const started = await next(e)
    const [usage, model] = await Promise.all([$.session.usage(), $.session.model()])
    await $.http.fetch(await $.env.get("HANDS_SESSION_START"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model, tokens: usage.context.tokens }),
    })
    return started
  })
}
