# pi-context-shake

A [pi](https://pi.dev) extension that frees context space by shaking heavy content
(tool output, images, thinking) out of the session history — like omp's `/shake`,
but for vanilla pi. It **rebuilds the session in place**: the session's `.jsonl`
file is rewritten atomically and pi reopens it (same code path as `/resume`), so
the persisted history, the transcript, and the in-memory agent state all reflect
the shaken history.

## Commands

| Command | Effect |
|---|---|
| `/shake` | Status: shaken modes, context usage, what can be removed |
| `/shake tools` | Elide big tool results / bash output / long text blocks in history |
| `/shake images` | Replace image blocks in history with one-line placeholders |
| `/shake thinking` | Drop thinking/reasoning blocks from history |
| `/shake all` | All of the above in one rebuild |

Thresholds (chars ≈ tokens/4): tool results / bash output > **2,000** chars become
a one-line placeholder with a 200-char head preview; text blocks > **12,000**
chars are trimmed to a 500-char head preview. Shaking a session again later
shakes the new turns too (idempotent on already-shaken content).

## How it works

1. The session file is read and its message entries are transformed: thinking
   dropped, tool results elided, images replaced, long text trimmed. Entry ids
   and the session tree are preserved.
2. The file is written atomically (temp + rename) with a single `pi-shake`
   state entry appended that records the enabled modes, then pi reopens the
   same file — the agent's in-memory history, the transcript, and the
   persisted session all become the shaken version.
3. A `context` hook (active while the session is marked shaken) also rewrites
   the copy sent to the provider on every LLM call, as a safety net for the
   brief window before/after the session refresh. Messages at or after the
   latest user message are never touched by the hook, so an in-flight turn
   keeps full context. The hook is idempotent: against already-shaken history
   it changes nothing.

**Footer note:** pi's footer context-usage figure is anchored to the
provider-reported usage of the *last* LLM call, so the freed space shows up
there after the next LLM call (any prompt). The transcript and the `/shake`
status update immediately.

Elision is **irreversible for the shaken session** — the original bytes are
gone from the file. If you might want them back, make a fork (`/tree`) before
shaking; forks made before the shake keep the full history.

**Thinking safety:** Anthropic requires signed thinking blocks in replayed
history. Signed/redacted thinking is only dropped when the active model is not
`anthropic-messages`, or when pi sends the `thinking-binding-controls` beta
(model with `supportsMidConvoEffort`). Otherwise it is kept and `/shake` status
notes it.

## Install

```sh
# from npm (listed in the pi.dev package catalog)
pi install npm:pi-context-shake

# or straight from git
pi install git:github.com/islamnofl/pi-context-shake

# try it for one run without installing
pi -e npm:pi-context-shake
```

Uninstall with `pi remove npm:pi-context-shake`.

## Develop / test

```sh
bun install
bunx tsc --noEmit   # typecheck against @earendil-works/pi-coding-agent
bun test.ts         # engine unit tests (message-level + entry-level rebuild)
```
