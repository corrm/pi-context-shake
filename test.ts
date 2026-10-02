/**
 * Engine tests for pi-shake. Run: bun test.ts
 */
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import extension, { shakeMessages, canDropSignedThinking, rebuildEntries, estimateMessageTokens, shouldSkipAutoCompaction } from "./index.ts";
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import type { ShakeMessage, ShakeModes, ShakeOptions, EntryLike } from "./index.ts";

const opts: ShakeOptions = { toolThreshold: 2000, blockThreshold: 12000, toolHead: 200, blockHead: 500 };
let failures = 0;

function check(name: string, cond: boolean, detail?: string) {
  if (cond) {
    console.log(`  ok  ${name}`);
  } else {
    failures++;
    console.log(`FAIL  ${name}${detail ? ` — ${detail}` : ""}`);
  }
}

const big = (n: number): string => "x".repeat(n);
const b64 = (n: number): string => Buffer.from(big(n)).toString("base64");

// --- Scenario 1: tool result elision + protection boundary ---------------
{
  const msgs: ShakeMessage[] = [
    { role: "user", content: "look at the log" },
    {
      role: "assistant",
      content: [
        { type: "toolCall", id: "c1", name: "bash", arguments: { command: "cat big.log" } },
      ],
    },
    { role: "toolResult", toolCallId: "c1", toolName: "bash", content: [{ type: "text", text: big(40_000) }], isError: false },
    { role: "user", content: "and this new image" },
    {
      role: "assistant",
      content: [{ type: "toolCall", id: "c2", name: "read", arguments: { path: "a.ts" } }],
    },
    { role: "toolResult", toolCallId: "c2", toolName: "read", content: [{ type: "text", text: big(60_000) }], isError: false },
  ];

  const all: ShakeModes = { tools: true, images: true, thinking: true };
  const { messages, stats } = shakeMessages(msgs, all, opts, true);

  const old = messages[2]!;
  check("old big tool result elided", typeof old.content === "object" && old.content!.length > 0 &&
    (old.content![0] as { text: string }).text.startsWith("[shaken: ~"), "got: " + JSON.stringify(old.content![0]));
  check("elided tool result keeps head preview", (old.content![0] as { text: string }).text.includes("(head) xxx"));
  check("tool stats counted", stats.toolCount === 1 && stats.toolChars > 30_000, `count=${stats.toolCount} chars=${stats.toolChars}`);

  const fresh = msgs[4]!;
  check("in-flight tool result protected", JSON.stringify(fresh.content) === JSON.stringify(msgs[4]!.content));
  const curUser = msgs[3]!;
  check("current user message protected", JSON.stringify(curUser) === JSON.stringify(msgs[3]!));
}

// --- Scenario 2: idle context — last user message is old, gets shaken -----
{
  const longPaste = big(20_000);
  const msgs: ShakeMessage[] = [
    { role: "user", content: longPaste },
    { role: "assistant", content: [{ type: "text", text: "done, here is the summary" }] },
  ];
  const { messages } = shakeMessages(msgs, { tools: true, images: false, thinking: false }, opts, true);
  const shaken = messages[0]!;
  check("old long user paste trimmed", typeof shaken.content === "string" &&
    shaken.content.includes("[shaken:") && (shaken.content as string).length < 2000,
    `len=${(shaken.content as string).length}`);
  check("final assistant reply untouched", messages[1]!.content === msgs[1]!.content);
}

// --- Scenario 3: images ---------------------------------------------------
{
  const msgs: ShakeMessage[] = [
    { role: "user", content: "describe" },
    { role: "assistant", content: [{ type: "text", text: "ok" }] },
    {
      role: "user",
      content: [
        { type: "text", text: "two shots:" },
        { type: "image", data: b64(100_000), mimeType: "image/png" },
        { type: "image", data: b64(50_000), mimeType: "image/jpeg" },
      ],
    },
    { role: "assistant", content: [{ type: "text", text: "final answer" }] },
  ];
  const { messages, stats } = shakeMessages(msgs, { tools: false, images: true, thinking: false }, opts, true);
  const shaken = messages[2]!;
  const texts = (shaken.content as { type: string; text?: string }[]).filter((b) => b.type === "text");
  const images = (shaken.content as { type: string }[]).filter((b) => b.type === "image");
  check("both images removed", images.length === 0);
  check("one placeholder for consecutive images", texts.some((t) => t.text!.startsWith("[shaken: image removed (image/png, ~98 KB)")),
    JSON.stringify(texts));
  check("image stats counted", stats.imageCount === 2 && stats.imageBytes > 150_000);
}

// --- Scenario 4: thinking, signed vs unsigned -----------------------------
{
  const msgs: ShakeMessage[] = [
    { role: "user", content: "think hard" },
    {
      role: "assistant",
      content: [
        { type: "thinking", thinking: big(5_000), thinkingSignature: "sig-abc" },
        { type: "thinking", thinking: big(3_000) },
        { type: "redactedThinking", data: "opaque" },
        { type: "text", text: "answer" },
      ],
    },
    { role: "user", content: "next" },
    { role: "assistant", content: [{ type: "text", text: "final" }] },
  ];

  const safe = shakeMessages(msgs, { tools: false, images: false, thinking: true }, opts, true);
  const safeBlocks = safe.messages[1]!.content as unknown as { type: string; text?: string; thinkingSignature?: string }[];
  check("unsigned thinking dropped", !safeBlocks.some((b) => b.type === "thinking"));
  check("redacted dropped when allowed", !safeBlocks.some((b) => b.type === "redactedThinking"));
  check("text kept", safeBlocks.some((b) => b.type === "text" && b.text === "answer"));
  check("thinking stats", safe.stats.thinkingCount === 3 && safe.stats.thinkingChars >= 8_000);

  const unsafe = shakeMessages(msgs, { tools: false, images: false, thinking: true }, opts, false);
  const unsafeBlocks = (unsafe.messages[1]!.content as { type: string; thinkingSignature?: string }[]);
  check("signed thinking kept when unsafe", unsafeBlocks.some((b) => b.type === "thinking" && b.thinkingSignature === "sig-abc"));
  check("redacted kept when unsafe", unsafeBlocks.some((b) => b.type === "redactedThinking"));
  check("unsigned still dropped when unsafe", !unsafeBlocks.some((b) => b.type === "thinking" && !b.thinkingSignature));
}

// --- Scenario 5: thinking-only assistant message doesn't become empty -----
{
  const msgs: ShakeMessage[] = [
    { role: "user", content: "think" },
    { role: "assistant", content: [{ type: "thinking", thinking: big(1_000) }] },
    { role: "user", content: "go" },
    { role: "assistant", content: [{ type: "text", text: "x" }] },
  ];
  const { messages } = shakeMessages(msgs, { tools: false, images: false, thinking: true }, opts, true);
  const content = messages[1]!.content as { type: string; text?: string }[];
  check("placeholder text inserted for emptied message", content.length === 1 && content[0]!.type === "text" &&
    content[0]!.text!.includes("thinking elided"));
}

// --- Scenario 6: bashExecution output elision ------------------------------
{
  const msgs: ShakeMessage[] = [
    { role: "user", content: "run it" },
    { role: "bashExecution", command: "make", output: big(30_000), exitCode: 0, cancelled: false, truncated: false },
    { role: "user", content: "now what?" },
    { role: "assistant", content: [{ type: "text", text: "built" }] },
  ];
  const { messages, stats } = shakeMessages(msgs, { tools: true, images: false, thinking: false }, opts, true);
  const bash = messages[1]!;
  check("bash output elided", typeof bash.output === "string" && bash.output.includes("[shaken:") && bash.output.length < 1500,
    `len=${bash.output?.length}`);
  check("bash stats", stats.bashChars > 25_000);
}

// --- Scenario 7: idempotency ------------------------------------------------
{
  const msgs: ShakeMessage[] = [
    { role: "user", content: "start" },
    {
      role: "assistant",
      content: [
        { type: "thinking", thinking: big(4_000) },
        { type: "toolCall", id: "c1", name: "bash", arguments: {} },
      ],
    },
    { role: "toolResult", toolCallId: "c1", toolName: "bash", content: [{ type: "text", text: big(50_000) }], isError: false },
    { role: "user", content: "again" },
    { role: "assistant", content: [{ type: "text", text: "final" }] },
  ];
  const all: ShakeModes = { tools: true, images: true, thinking: true };
  const pass1 = shakeMessages(msgs, all, opts, true);
  const pass2 = shakeMessages(pass1.messages, all, opts, true);
  const noChange = pass2.messages.every((m, i) => JSON.stringify(m) === JSON.stringify(pass1.messages[i]));
  const secondFreed =
    pass2.stats.toolChars + pass2.stats.bashChars + pass2.stats.blockChars + pass2.stats.imageBytes + pass2.stats.thinkingChars;
  check("second pass is a no-op", noChange && secondFreed === 0, `freed=${secondFreed}`);
}

// --- Scenario 8: everything off → untouched ---------------------------------
{
  const msgs: ShakeMessage[] = [
    { role: "user", content: big(50_000) },
    { role: "assistant", content: [{ type: "thinking", thinking: big(50_000) }, { type: "text", text: "hi" }] },
  ];
  const off: ShakeModes = { tools: false, images: false, thinking: false };
  const { messages, stats } = shakeMessages(msgs, off, opts, true);
  const identical = messages.every((m, i) => m === msgs[i]);
  const freed = stats.toolChars + stats.bashChars + stats.blockChars + stats.imageBytes + stats.thinkingChars;
  check("off policy leaves messages identical", identical && freed === 0);
}

// --- Scenario 9: canDropSignedThinking --------------------------------------
{
  check("non-anthropic allowed", canDropSignedThinking({ api: "openai-completions" }) === true);
  check("anthropic without beta kept", canDropSignedThinking({ api: "anthropic-messages" }) === false);
  check("anthropic with beta allowed", canDropSignedThinking({ api: "anthropic-messages", compat: { supportsMidConvoEffort: true } }) === true);
  check("unknown model allowed", canDropSignedThinking(undefined) === true);
}

// --- Scenario 10: compaction/branch summary roles pass through -------------
{
  const msgs: ShakeMessage[] = [
    { role: "compactionSummary", content: "old summary" },
    { role: "branchSummary", content: "branch summary text" },
    { role: "user", content: "hi" },
    { role: "assistant", content: [{ type: "text", text: "yo" }] },
  ];
  const { messages, stats } = shakeMessages(msgs, { tools: true, images: true, thinking: true }, opts, true);
  const identical = messages.every((m, i) => m === msgs[i]);
  check("summary roles untouched", identical && stats.thinkingCount === 0);
}


// --- Scenario 11: rebuildEntries (session file rebuild) ---------------------
{
  const header: EntryLike = { type: "session", version: 3, id: "abc", created_at: 1, updated_at: 2, cwd: "/tmp" };
  const e1: EntryLike = { type: "message", id: "1", parentId: null, timestamp: 3, message: { role: "user", content: big(20_000) } };
  const e2: EntryLike = {
    type: "message",
    id: "2",
    parentId: "1",
    timestamp: 4,
    message: {
      role: "assistant",
      content: [
        { type: "thinking", thinking: big(5_000), thinkingSignature: "sig" },
        { type: "toolCall", id: "c1", name: "bash", arguments: {} },
      ],
    },
  };
  const e3: EntryLike = {
    type: "message",
    id: "3",
    parentId: "2",
    timestamp: 5,
    message: { role: "toolResult", toolCallId: "c1", toolName: "bash", isError: false, content: [{ type: "text", text: big(40_000) }] },
  };
  const e4: EntryLike = {
    type: "message",
    id: "4",
    parentId: "3",
    timestamp: 6,
    message: {
      role: "user",
      content: [
        { type: "text", text: "see:" },
        { type: "image", data: b64(100_000), mimeType: "image/png" },
      ],
    },
  };
  const e5: EntryLike = {
    type: "message",
    id: "5",
    parentId: "4",
    timestamp: 7,
    message: { role: "bashExecution", command: "make", output: big(30_000), exitCode: 0, cancelled: false, truncated: false },
  };
  const e6: EntryLike = { type: "custom_message", id: "6", parentId: "5", timestamp: 8, content: big(20_000) };
  const e7: EntryLike = { type: "label", id: "7", parentId: "6", timestamp: 9, marker: "m", label: "keep" };
  const e8: EntryLike = {
    type: "message",
    id: "8",
    parentId: "7",
    timestamp: 10,
    message: { role: "assistant", content: [{ type: "thinking", thinking: big(4_000) }, { type: "text", text: "final" }] },
  };
  const entries: EntryLike[] = [header, e1, e2, e3, e4, e5, e6, e7, e8];
  const all: ShakeModes = { tools: true, images: true, thinking: true };

  const { entries: rebuilt, stats } = rebuildEntries(entries, opts, all, true);
  const r1 = rebuilt[1] as unknown as { message: { content: string } };
  const r2 = rebuilt[2] as unknown as { message: { content: { type: string; thinkingSignature?: string }[] } };
  const r3 = rebuilt[3] as unknown as { message: { content: { type: string; text?: string }[] } };
  const r4 = rebuilt[4] as unknown as { message: { content: { type: string; text?: string }[] } };
  const r5 = rebuilt[5] as unknown as { message: { output: string } };
  const r6 = rebuilt[6] as unknown as { content: string };

  check("long user string elided", typeof r1.message.content === "string" && r1.message.content.includes("[shaken:") && r1.message.content.length < 2000);
  check("assistant thinking dropped, toolCall kept", r2.message.content.every((b) => b.type !== "thinking") && r2.message.content.some((b) => b.type === "toolCall"));
  check("tool result elided", r3.message.content.length === 1 && r3.message.content[0]!.text!.startsWith("[shaken: ~") && r3.message.content[0]!.text!.includes('"bash"'));
  check(
    "image replaced with placeholder",
    r4.message.content.every((b) => b.type !== "image") &&
      r4.message.content.some((b) => b.type === "text" && b.text!.startsWith("[shaken: image removed (image/png, ~98 KB)")),
  );
  check("bash output elided", r5.message.output.includes("[shaken:") && r5.message.output.length < 1500);
  check("custom_message long string elided", typeof r6.content === "string" && r6.content.length < 2000 && r6.content.includes("[shaken:"));
  check("header/label entries untouched", rebuilt[0] === header && rebuilt[7] === e7);
  check("rebuild counted everything", stats.thinkingCount === 2 && stats.toolCount >= 2 && stats.imageCount === 1 && stats.blockCount >= 2);

  const pass2 = rebuildEntries(rebuilt, opts, all, true);
  const freed2 = pass2.stats.toolChars + pass2.stats.bashChars + pass2.stats.blockChars + pass2.stats.imageBytes + pass2.stats.thinkingChars;
  check("second rebuild is a no-op", pass2.entries.every((e, i) => e === rebuilt[i]) && freed2 === 0);

  const keptSigned = rebuildEntries(entries, opts, { tools: false, images: false, thinking: true }, false);
  const k2 = keptSigned.entries[2] as unknown as { message: { content: { type: string; thinkingSignature?: string }[] } };
  const k8 = keptSigned.entries[8] as unknown as { message: { content: { type: string }[] } };
  check("signed thinking kept when unsafe", k2.message.content.some((b) => b.type === "thinking" && b.thinkingSignature === "sig"));
  check("unsigned thinking dropped even when unsafe", k8.message.content.every((b) => b.type !== "thinking"));

  const off = rebuildEntries(entries, opts, { tools: false, images: false, thinking: false }, true);
  check("off modes leave entries identical", off.entries.every((e, i) => e === entries[i]));
}

// --- Scenario 12: estimateMessageTokens -----------------------------------
{
  const msgs: ShakeMessage[] = [
    { role: "user", content: "x".repeat(4000) },
    {
      role: "assistant",
      content: [
        { type: "thinking", thinking: "t".repeat(4000) },
        { type: "toolCall", id: "c1", name: "bash", arguments: { command: "ls" } },
      ],
    },
    { role: "toolResult", toolCallId: "c1", toolName: "bash", isError: false, content: [{ type: "text", text: "y".repeat(8000) }] },
    { role: "bashExecution", command: "ls", output: "z".repeat(12000), exitCode: 0, cancelled: false, truncated: false },
    { role: "compactionSummary", content: "s".repeat(4000) },
  ];
  const est = estimateMessageTokens(msgs);
  const user = 4000 / 4;
  const assistant = Math.ceil((4000 + ("bash".length + JSON.stringify({ command: "ls" }).length)) / 4);
  const tool = 8000 / 4;
  const bash = Math.ceil(("ls".length + 12000) / 4);
  const compaction = 4000 / 4;
  check("per-message estimator sums correctly", est === user + assistant + tool + bash + compaction, `got=${est} want=${user + assistant + tool + bash + compaction}`);
}

// --- Scenario 13: shouldSkipAutoCompaction ----------------------------------
{
  const window = 200_000;
  const reserve = 10_000;
  const all: ShakeModes = { tools: true, images: true, thinking: true };

  // Big history that is shaken down well under the threshold.
  const bigHistory: ShakeMessage[] = [
    { role: "user", content: "look" },
    { role: "assistant", content: [{ type: "thinking", thinking: big(50_000) }] },
    { role: "assistant", content: [{ type: "toolCall", id: "c1", name: "bash", arguments: {} }] },
    { role: "toolResult", toolCallId: "c1", toolName: "bash", isError: false, content: [{ type: "text", text: big(80_000) }] },
    { role: "user", content: "go" },
    { role: "assistant", content: [{ type: "text", text: "done" }] },
  ];
  check("skip compaction when shaken fits", shouldSkipAutoCompaction({ modes: all, messages: bigHistory, opts, contextWindow: window, reserveTokens: reserve, model: undefined }) === true);

  // Still too big: the latest user message (and its pending turn) is a giant
  // paste — boundary-protected, so shaking cannot reduce it.
  const stillBig: ShakeMessage[] = [
    { role: "user", content: "hi" },
    { role: "assistant", content: [{ type: "text", text: "ok" }] },
    { role: "user", content: big(1_000_000) },
    { role: "assistant", content: [{ type: "text", text: "working" }, { type: "toolCall", id: "c9", name: "bash", arguments: {} }] },
  ];
  check("do not skip when boundary-protected content is over threshold", shouldSkipAutoCompaction({ modes: all, messages: stillBig, opts, contextWindow: window, reserveTokens: reserve, model: undefined }) === false);

  check("never skip when modes off", shouldSkipAutoCompaction({ modes: { tools: false, images: false, thinking: false }, messages: bigHistory, opts, contextWindow: window, reserveTokens: reserve, model: undefined }) === false);
  check("never skip with no context window", shouldSkipAutoCompaction({ modes: all, messages: bigHistory, opts, contextWindow: 0, reserveTokens: reserve, model: undefined }) === false);
}

// --- Scenario 14: mid-turn shakes run once after full settlement ------------
{
  let command: Parameters<ExtensionAPI["registerCommand"]>[1] | undefined;
  const handlers = new Map<string, (event: never, ctx: ExtensionCommandContext) => unknown>();
  extension({
    on: (name: string, handler: (event: never, ctx: ExtensionCommandContext) => unknown) => handlers.set(name, handler),
    registerCommand: (_name: string, registered: NonNullable<typeof command>) => { command = registered; },
  } as unknown as ExtensionAPI);
  const dir = mkdtempSync(join(tmpdir(), "pi-shake-test-"));
  const file = join(dir, "session.jsonl");
  const originalEntries: EntryLike[] = [
    { type: "session", version: 3, id: "session" },
    { type: "message", id: "u", parentId: null, message: { role: "user", content: "look" } },
    { type: "message", id: "a", parentId: "u", message: { role: "assistant", content: [{ type: "toolCall", id: "c", name: "read", arguments: {} }] } },
    { type: "message", id: "t", parentId: "a", message: { role: "toolResult", toolCallId: "c", toolName: "read", content: [{ type: "text", text: big(40_000) }, { type: "image", data: b64(1000), mimeType: "image/png" }] } },
  ];
  const original = originalEntries.map((e) => JSON.stringify(e)).join("\n") + "\n";
  let entries = originalEntries;
  let idle = false;
  let rereads = 0;
  const notices: { text: string; type: string }[] = [];
  const ctx = {
    isIdle: () => idle,
    hasUI: false,
    getContextUsage: () => undefined,
    sessionManager: {
      getEntries: () => entries,
      getSessionFile: () => file,
      getLeafId: () => "t",
      setSessionFile: (path: string) => {
        check("re-read only happens while idle", idle);
        rereads++;
        entries = readFileSync(path, "utf8").trim().split("\n").map((line) => JSON.parse(line));
      },
    },
    ui: {
      notify: (text: string, type: string) => notices.push({ text, type }),
      setWidget: () => {},
    },
  } as unknown as ExtensionCommandContext;
  try {
    writeFileSync(file, original);
    await command!.handler("", ctx);
    check("bare shake still shows status while busy", notices[0]?.text.includes("removable now:") === true);
    idle = true;
    await handlers.get("agent_settled")?.({} as never, ctx);
    check("showing status does not queue a rebuild", rereads === 0 && readFileSync(file, "utf8") === original);
    idle = false;
    notices.length = 0;
    for (const mode of ["tools", "images", "tools"]) await command!.handler(mode, ctx);
    check("busy shakes acknowledge queue instead of warning", notices.length === 3 && notices.every((n) => n.type === "info" && n.text.includes("queued")));
    check("busy shakes leave disk and live history untouched", readFileSync(file, "utf8") === original && rereads === 0);
    await handlers.get("agent_end")?.({} as never, ctx);
    await handlers.get("agent_settled")?.({} as never, ctx);
    check("agent_end and a busy boundary do not drain the queue", readFileSync(file, "utf8") === original && rereads === 0);
    // Simulate a retry/continuation finishing after agent_end.
    const finalEntry: EntryLike = { type: "message", id: "final", parentId: "t", message: { role: "assistant", content: [{ type: "thinking", thinking: big(4000) }, { type: "text", text: "done" }] } };
    writeFileSync(file, original + JSON.stringify(finalEntry) + "\n");
    entries = [...originalEntries, finalEntry];
    await command!.handler("thinking", ctx);
    idle = true;
    await handlers.get("agent_settled")?.({} as never, ctx);
    const state = entries.find((e) => e.customType === "pi-shake")?.data as { modes?: ShakeModes } | undefined;
    check("queued requests merge all requested modes into one rebuild", rereads === 1 && state?.modes?.tools === true && state.modes.images && state.modes.thinking);
    const shaken = readFileSync(file, "utf8");
    check("queued rebuild includes the completed continuation", shaken.includes("[shaken:") && !shaken.includes('"type":"image"') && !shaken.includes('"type":"thinking"') && shaken.includes('"id":"final"'));
    notices.length = 0;
    await handlers.get("agent_settled")?.({} as never, ctx);
    check("settlement drains pending shake exactly once", rereads === 1 && notices.length === 0);
    await command!.handler("all", ctx);
    check("idle shake still executes immediately", notices[0]?.text === "Nothing to shake in history.");
    for (const event of ["session_shutdown", "session_start"]) {
      writeFileSync(file, original);
      entries = originalEntries;
      idle = false;
      await command!.handler("all", ctx);
      await handlers.get(event)?.({} as never, ctx);
      idle = true;
      await handlers.get("agent_settled")?.({} as never, ctx);
      check(`${event} clears queued shakes`, readFileSync(file, "utf8") === original && rereads === 1);
    }
    idle = false;
    await command!.handler("tools", ctx);
    writeFileSync(file, "not json\n");
    notices.length = 0;
    idle = true;
    await handlers.get("agent_settled")?.({} as never, ctx);
    check("queued rebuild reports file errors without modifying history", notices[0]?.type === "error" && notices[0].text.includes("unparseable") && readFileSync(file, "utf8") === "not json\n");
    notices.length = 0;
    await handlers.get("agent_settled")?.({} as never, ctx);
    check("failed queued rebuild is not retried on every settlement", notices.length === 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}
console.log(failures === 0 ? "\nall tests passed" : `\n${failures} test(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
