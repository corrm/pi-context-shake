/**
 * pi-shake — shake heavy content out of the session history to free space.
 *
 * Commands:
 *   /shake              status: shaken modes, context usage, what can be removed
 *   /shake tools        rebuild history in place: elide big tool results,
 *                       bash output, and long text blocks
 *   /shake images       rebuild history in place: replace image blocks with
 *                       one-line placeholders
 *   /shake thinking     rebuild history in place: drop thinking blocks
 *   /shake all          rebuild with every mode at once
 *
 * How it works:
 *   The session's .jsonl file is rewritten in place (atomically) and the
 *   live session manager re-reads it, so the persisted session reflects the
 *   shaken history — like omp's /shake.
 *
 *   The running process keeps its state (no session switch). A `context`
 *   hook (active while the session is marked shaken) rewrites the copy of
 *   the history sent to the provider on every LLM call, so the live
 *   process also only sends the shaken history until the session is
 *   reloaded. Messages at or after the latest user message are never
 *   shaken by the hook, so an in-flight turn keeps full context. The hook
 *   is idempotent: against already-shaken history it changes nothing.
 *
 *   The footer's context-usage figure is anchored to the last
 *   provider-reported call and refreshes after the next LLM call.
 *
 *   Elision is irreversible for the shaken session (the original bytes are
 *   gone from the file) — the pre-shake history survives in any fork made
 *   before the shake.
 *
 *   Thinking safety: Anthropic requires signed thinking blocks in replayed
 *   history. Signed/redacted thinking is only dropped when the active model
 *   is not anthropic-messages, or when pi sends the thinking-binding-controls
 *   beta (model with supportsMidConvoEffort). Otherwise it is kept and a note
 *   is shown in /shake status.
 */

import { readFileSync, writeFileSync, renameSync } from "node:fs";
import { randomUUID } from "node:crypto";
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";

// ---------------------------------------------------------------------------
// Local structural types — only the type imports above are erased at load.
// ---------------------------------------------------------------------------
interface TextBlock {
  type: "text";
  text: string;
}

interface ImageBlock {
  type: "image";
  data: string;
  mimeType: string;
}

interface ThinkingBlock {
  type: "thinking";
  thinking: string;
  thinkingSignature?: string;
  redacted?: boolean;
}

type LegacyRedactedThinkingBlock = { type: "redactedThinking"; data?: string };

type Block =
  | TextBlock
  | ImageBlock
  | ThinkingBlock
  | LegacyRedactedThinkingBlock
  | { type: string; [key: string]: unknown };

/** Loosen shape of a context message; everything else passes through untouched. */
export interface ShakeMessage {
  role: string;
  content?: string | Block[];
  toolName?: string;
  output?: string;
  [key: string]: unknown;
}

export interface ShakeModes {
  tools: boolean;
  images: boolean;
  thinking: boolean;
}

export interface ShakeOptions {
  /** Tool results / bash output longer than this (chars) get elided. */
  toolThreshold: number;
  /** Text blocks longer than this (chars) get trimmed. */
  blockThreshold: number;
  /** Head preview kept when eliding tool results (chars). */
  toolHead: number;
  /** Head preview kept when trimming long text blocks (chars). */
  blockHead: number;
}

export interface ShakeStats {
  toolChars: number;
  bashChars: number;
  blockChars: number;
  toolCount: number;
  blockCount: number;
  imageBytes: number;
  imageCount: number;
  thinkingChars: number;
  thinkingCount: number;
}

/** One parsed session-file entry (shape checked per use). */
export type EntryLike = { type: string; [key: string]: unknown };

const STATE_TYPE = "pi-shake";
const DEFAULT_OPTIONS: ShakeOptions = {
  toolThreshold: 2000,
  blockThreshold: 12000,
  toolHead: 200,
  blockHead: 500,
};
const KINDS = ["tools", "images", "thinking", "all"] as const;
type ModelLike = { api?: string; compat?: { supportsMidConvoEffort?: boolean } } | undefined;

const estTokens = (chars: number): number => Math.max(0, Math.round(chars / 4));
const fmt = (n: number): string => n.toLocaleString("en-US");
const isTextBlock = (b: Block): b is TextBlock => b.type === "text";
const isImageBlock = (b: Block): b is ImageBlock => b.type === "image";
const isThinkingBlock = (b: Block): b is ThinkingBlock => b.type === "thinking";
const textLen = (b: Block): number => (isTextBlock(b) ? b.text.length : 0);
const freedChars = (s: ShakeStats): number => s.toolChars + s.bashChars + s.blockChars + s.imageBytes + s.thinkingChars;

const freshStats = (): ShakeStats => ({
  toolChars: 0,
  bashChars: 0,
  blockChars: 0,
  toolCount: 0,
  blockCount: 0,
  imageBytes: 0,
  imageCount: 0,
  thinkingChars: 0,
  thinkingCount: 0,
});

// ---------------------------------------------------------------------------
// Block-level transforms (shared by the runtime hook and the file rebuild)
// ---------------------------------------------------------------------------

/** Drop thinking/reasoning blocks; signed only when dropSigned is true. */
function shakeThinkingBlocks(content: Block[], dropSigned: boolean): { content: Block[]; chars: number; count: number; changed: boolean } {
  let chars = 0;
  let count = 0;
  let changed = false;
  const kept: Block[] = [];
  for (const b of content) {
    if (!isThinkingBlock(b) && b.type !== "redactedThinking") {
      kept.push(b);
      continue;
    }
    const signed = isThinkingBlock(b) && typeof b.thinkingSignature === "string" && b.thinkingSignature.trim().length > 0;
    const redacted = isThinkingBlock(b) ? b.redacted === true : true;
    if (!dropSigned && (signed || redacted)) {
      kept.push(b);
      continue;
    }
    count++;
    chars += isThinkingBlock(b) ? b.thinking.length : 0;
    changed = true;
  }
  if (changed && kept.length === 0) kept.push({ type: "text", text: "(thinking elided by /shake)" });
  return { content: changed ? kept : content, chars, count, changed };
}

/** Replace image blocks with one-line placeholders (consecutive → one note). */
function shakeImageBlocks(content: Block[]): { content: Block[]; bytes: number; count: number; changed: boolean } {
  let bytes = 0;
  let count = 0;
  let changed = false;
  const replaced: Block[] = [];
  let prevImagePlaceholder = false;
  for (const b of content) {
    if (isImageBlock(b)) {
      const dataLen = typeof b.data === "string" ? b.data.length : 0;
      count++;
      bytes += dataLen;
      if (!prevImagePlaceholder) {
        const kb = Math.round((dataLen * 0.75) / 1024);
        replaced.push({ type: "text", text: `[shaken: image removed (${b.mimeType ?? "image"}, ~${kb} KB)]` });
      }
      prevImagePlaceholder = true;
      changed = true;
      continue;
    }
    replaced.push(b);
    prevImagePlaceholder = isTextBlock(b) && b.text.startsWith("[shaken: image removed");
  }
  return { content: changed ? replaced : content, bytes, count, changed };
}

/** Elide a whole tool result's text to a placeholder with a head preview. */
function elideToolResultContent(content: Block[], opts: ShakeOptions, toolName: string): { content: Block[]; chars: number; changed: boolean } {
  const textChars = content.reduce((n, b) => n + textLen(b), 0);
  if (textChars <= opts.toolThreshold) return { content, chars: 0, changed: false };
  let head = "";
  for (const b of content) {
    if (isTextBlock(b) && b.text.length > 0) {
      head = b.text.slice(0, opts.toolHead).trim();
      break;
    }
  }
  const placeholder =
    `[shaken: ~${estTokens(textChars)} tokens elided from "${toolName}" result]` +
    (head ? `\n(head) ${head.replace(/\s*\n\s*/g, " ").slice(0, opts.toolHead)}` : "");
  const kept = content.filter((b) => textLen(b) === 0);
  return { content: [{ type: "text", text: placeholder }, ...kept], chars: Math.max(0, textChars - placeholder.length), changed: true };
}

/** Trim text blocks longer than blockThreshold to a head preview. */
function elideLongTextBlocks(content: Block[], opts: ShakeOptions): { content: Block[]; chars: number; count: number; changed: boolean } {
  let chars = 0;
  let count = 0;
  let changed = false;
  const trimmed: Block[] = [];
  for (const b of content) {
    if (isTextBlock(b) && b.text.length > opts.blockThreshold) {
      const elided = b.text.length - opts.blockHead;
      const placeholder = `${b.text.slice(0, opts.blockHead)}\n… [shaken: ~${estTokens(elided)} tokens elided from long text]`;
      count++;
      chars += Math.max(0, b.text.length - placeholder.length);
      changed = true;
      trimmed.push({ type: "text", text: placeholder });
    } else {
      trimmed.push(b);
    }
  }
  return { content: changed ? trimmed : content, chars, count, changed };
}

/** Trim a long plain string to a head preview. */
function elideLongString(text: string, opts: ShakeOptions): { text: string; chars: number; changed: boolean } {
  if (text.length <= opts.blockThreshold) return { text, chars: 0, changed: false };
  const elided = text.length - opts.blockHead;
  const placeholder = `${text.slice(0, opts.blockHead)}\n… [shaken: ~${estTokens(elided)} tokens elided from long text]`;
  return { text: placeholder, chars: Math.max(0, text.length - placeholder.length), changed: true };
}

// ---------------------------------------------------------------------------
// Message-level engine (runtime `context` hook)
// ---------------------------------------------------------------------------

/**
 * Only messages strictly before this index may be shaken by the runtime
 * hook. The latest user message is always protected (it may be the prompt
 * currently being processed). If the conversation already produced a final
 * assistant reply after it (no pending tool calls), the user message itself
 * is old too and may be shaken.
 */
function boundaryIndex(messages: ShakeMessage[]): number {
  let lastUser = -1;
  for (let i = 0; i < messages.length; i++) {
    if (messages[i]!.role === "user") lastUser = i;
  }
  if (lastUser < 0) return -1;
  const last = messages[messages.length - 1];
  const trailingIsFinal =
    last !== undefined &&
    last.role === "assistant" &&
    Array.isArray(last.content) &&
    !last.content.some((b) => b.type === "toolCall");
  return trailingIsFinal ? lastUser + 1 : lastUser;
}

/**
 * Apply shake modes to a list of context messages (pure; input not mutated).
 * Used by the runtime hook: only history before the active turn is touched.
 */
export function shakeMessages(
  messages: ShakeMessage[],
  policy: ShakeModes,
  opts: ShakeOptions,
  dropSignedThinking: boolean,
): { messages: ShakeMessage[]; stats: ShakeStats } {
  const stats = freshStats();
  const boundary = boundaryIndex(messages);
  const out: ShakeMessage[] = new Array(messages.length);

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]!;
    if (i >= boundary) {
      out[i] = msg;
      continue;
    }
    let next: ShakeMessage = msg;
    let changed = false;

    if (policy.thinking && next.role === "assistant" && Array.isArray(next.content)) {
      const r = shakeThinkingBlocks(next.content, dropSignedThinking);
      if (r.changed) {
        next = { ...next, content: r.content };
        changed = true;
        stats.thinkingChars += r.chars;
        stats.thinkingCount += r.count;
      }
    }
    if (policy.images && next.content !== undefined && Array.isArray(next.content)) {
      const r = shakeImageBlocks(next.content);
      if (r.changed) {
        next = { ...next, content: r.content };
        changed = true;
        stats.imageBytes += r.bytes;
        stats.imageCount += r.count;
      }
    }
    if (policy.tools && next.role === "toolResult" && Array.isArray(next.content)) {
      const r = elideToolResultContent(next.content, opts, next.toolName ?? "tool");
      if (r.changed) {
        next = { ...next, content: r.content };
        changed = true;
        stats.toolCount++;
        stats.toolChars += r.chars;
      }
    }
    if (policy.tools && next.role === "bashExecution" && typeof next.output === "string" && next.output.length > opts.toolThreshold) {
      const elided = next.output.length - opts.toolHead;
      next = { ...next, output: `${next.output.slice(0, opts.toolHead).trimEnd()}\n… [shaken: ~${estTokens(elided)} tokens elided from bash output]` };
      changed = true;
      stats.toolCount++;
      stats.bashChars += elided;
    }
    if (policy.tools && (next.role === "user" || next.role === "assistant" || next.role === "custom")) {
      if (typeof next.content === "string") {
        const r = elideLongString(next.content, opts);
        if (r.changed) {
          next = { ...next, content: r.text };
          changed = true;
          stats.blockCount++;
          stats.blockChars += r.chars;
        }
      } else if (Array.isArray(next.content)) {
        const r = elideLongTextBlocks(next.content, opts);
        if (r.changed) {
          next = { ...next, content: r.content };
          changed = true;
          stats.blockCount += r.count;
          stats.blockChars += r.chars;
        }
      }
    }

    out[i] = changed ? next : msg;
  }

  return { messages: out, stats };
}

// ---------------------------------------------------------------------------
// Entry-level engine (session-file rebuild)
// ---------------------------------------------------------------------------

/**
 * Transform parsed session-file entries: every message entry on every branch
 * is shaken (no live-turn boundary — the turns are finished once the file is
 * written). Untouched entries are returned by reference.
 */
export function rebuildEntries(
  entries: EntryLike[],
  opts: ShakeOptions,
  modes: ShakeModes,
  dropSignedThinking: boolean,
): { entries: EntryLike[]; stats: ShakeStats } {
  const stats = freshStats();

  const out: EntryLike[] = entries.map((entry) => {
    if (entry.type === "message") {
      const msg = entry.message as ShakeMessage | undefined;
      if (!msg || typeof msg !== "object") return entry;
      let next: ShakeMessage = msg;
      let changed = false;

      if (modes.thinking && next.role === "assistant" && Array.isArray(next.content)) {
        const r = shakeThinkingBlocks(next.content, dropSignedThinking);
        if (r.changed) {
          next = { ...next, content: r.content };
          changed = true;
          stats.thinkingChars += r.chars;
          stats.thinkingCount += r.count;
        }
      }
      if (modes.images && next.content !== undefined && Array.isArray(next.content)) {
        const r = shakeImageBlocks(next.content);
        if (r.changed) {
          next = { ...next, content: r.content };
          changed = true;
          stats.imageBytes += r.bytes;
          stats.imageCount += r.count;
        }
      }
      if (modes.tools && next.role === "toolResult" && Array.isArray(next.content)) {
        const r = elideToolResultContent(next.content, opts, next.toolName ?? "tool");
        if (r.changed) {
          next = { ...next, content: r.content };
          changed = true;
          stats.toolCount++;
          stats.toolChars += r.chars;
        }
      }
      if (modes.tools && next.role === "bashExecution" && typeof next.output === "string" && next.output.length > opts.toolThreshold) {
        const elided = next.output.length - opts.toolHead;
        next = { ...next, output: `${next.output.slice(0, opts.toolHead).trimEnd()}\n… [shaken: ~${estTokens(elided)} tokens elided from bash output]` };
        changed = true;
        stats.toolCount++;
        stats.bashChars += elided;
      }
      if (modes.tools && (next.role === "user" || next.role === "assistant" || next.role === "custom")) {
        if (typeof next.content === "string") {
          const r = elideLongString(next.content, opts);
          if (r.changed) {
            next = { ...next, content: r.text };
            changed = true;
            stats.blockCount++;
            stats.blockChars += r.chars;
          }
        } else if (Array.isArray(next.content)) {
          const r = elideLongTextBlocks(next.content, opts);
          if (r.changed) {
            next = { ...next, content: r.content };
            changed = true;
            stats.blockCount += r.count;
            stats.blockChars += r.chars;
          }
        }
      }

      return changed ? { ...entry, message: next } : entry;
    }

    if (entry.type === "custom_message") {
      const content = entry.content;
      let next: EntryLike = entry;
      let changed = false;

      if (modes.images && Array.isArray(content)) {
        const r = shakeImageBlocks(content as Block[]);
        if (r.changed) {
          next = { ...next, content: r.content };
          changed = true;
          stats.imageBytes += r.bytes;
          stats.imageCount += r.count;
        }
      }
      if (modes.tools) {
        if (typeof next.content === "string") {
          const r = elideLongString(next.content, opts);
          if (r.changed) {
            next = { ...next, content: r.text };
            changed = true;
            stats.blockCount++;
            stats.blockChars += r.chars;
          }
        } else if (Array.isArray(next.content)) {
          const r = elideLongTextBlocks(next.content as Block[], opts);
          if (r.changed) {
            next = { ...next, content: r.content };
            changed = true;
            stats.blockCount += r.count;
            stats.blockChars += r.chars;
          }
        }
      }

      return changed ? next : entry;
    }

    return entry;
  });

  return { entries: out, stats };
}

/**
 * Signed thinking blocks (Anthropic) must normally be replayed verbatim in
 * multi-turn requests. Dropping them is only safe when the active model is
 * not anthropic-messages, or when pi sends the thinking-binding-controls
 * beta (supportsMidConvoEffort), which makes the API drop invalid thinking
 * instead of rejecting the request.
 */
export function canDropSignedThinking(model: ModelLike): boolean {
  if (!model || model.api !== "anthropic-messages") return true;
  return model.compat?.supportsMidConvoEffort === true;
}

// ---------------------------------------------------------------------------
// Extension factory
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
  let modes: ShakeModes = { tools: false, images: false, thinking: false };
  let opts: ShakeOptions = { ...DEFAULT_OPTIONS };

  // Reconstruct shaken-modes from the session (last pi-shake entry wins).
  pi.on("session_start", async (_event, ctx) => {
    modes = { tools: false, images: false, thinking: false };
    opts = { ...DEFAULT_OPTIONS };
    for (const entry of ctx.sessionManager.getEntries()) {
      if (entry.type !== "custom" || entry.customType !== STATE_TYPE) continue;
      const data = entry.data;
      if (!data || typeof data !== "object" || Array.isArray(data)) continue;
      if (!("modes" in data) && !("opts" in data)) continue;
      // Written by this extension in session format { modes, opts }; validate the keys we read.
      const stateData = data as { modes?: Partial<ShakeModes>; opts?: Partial<ShakeOptions> };
      modes = { tools: !!stateData.modes?.tools, images: !!stateData.modes?.images, thinking: !!stateData.modes?.thinking };
      opts = { ...DEFAULT_OPTIONS, ...(stateData.opts ?? {}) };
    }
  });

  // Keep the live in-memory message list slim until the session is re-read.
  pi.on("context", async (event, ctx) => {
    if (!modes.tools && !modes.images && !modes.thinking) return;
    const messages = event.messages as unknown as ShakeMessage[];
    if (!Array.isArray(messages) || messages.length === 0) return;
    const model = ctx.model as unknown as ModelLike;
    const { messages: shaken, stats } = shakeMessages(messages, modes, opts, canDropSignedThinking(model));
    if (freedChars(stats) === 0) return;
    return { messages: shaken as unknown as typeof event.messages };
  });

  const estimateRemovable = (ctx: ExtensionCommandContext, wanted: ShakeModes): ShakeStats => {
    const entries = ctx.sessionManager.getEntries() as unknown as EntryLike[];
    const model = ctx.model as unknown as ModelLike;
    return rebuildEntries(entries, opts, wanted, canDropSignedThinking(model)).stats;
  };

  const rebuildHistory = async (ctx: ExtensionCommandContext, wanted: ShakeModes): Promise<void> => {
    if (!ctx.isIdle()) {
      ctx.ui.notify("shake: wait for the agent to finish before rebuilding history", "warning");
      return;
    }
    const file = ctx.sessionManager.getSessionFile();
    if (!file) {
      ctx.ui.notify("shake: session is not saved to disk (ephemeral)", "error");
      return;
    }

    let lines: string[];
    try {
      lines = readFileSync(file, "utf8").split("\n").filter((l) => l.trim().length > 0);
    } catch (err) {
      ctx.ui.notify(`shake: cannot read session file: ${err instanceof Error ? err.message : String(err)}`, "error");
      return;
    }
    let entries: EntryLike[];
    try {
      entries = lines.map((l) => JSON.parse(l) as EntryLike);
    } catch (err) {
      ctx.ui.notify(`shake: session file has unparseable lines: ${err instanceof Error ? err.message : String(err)}`, "error");
      return;
    }

    const model = ctx.model as unknown as ModelLike;
    const { entries: rebuilt, stats } = rebuildEntries(entries, opts, wanted, canDropSignedThinking(model));
    if (freedChars(stats) === 0) {
      ctx.ui.notify("Nothing to shake in history.", "info");
      return;
    }
    const merged: ShakeModes = {
      tools: modes.tools || wanted.tools,
      images: modes.images || wanted.images,
      thinking: modes.thinking || wanted.thinking,
    };

    // Normalize state: exactly one pi-shake entry, chained onto the current leaf.
    const withoutState = rebuilt.filter((e) => !(e.type === "custom" && e.customType === STATE_TYPE));
    const usedIds = new Set<string>();
    for (const e of withoutState) if (typeof e.id === "string") usedIds.add(e.id);
    let stateId = "";
    for (let i = 0; i < 100 && !stateId; i++) {
      const candidate = randomUUID().slice(0, 8);
      if (!usedIds.has(candidate)) stateId = candidate;
    }
    if (!stateId) {
      ctx.ui.notify("shake: could not allocate a unique entry id", "error");
      return;
    }
    const stateEntry: EntryLike = {
      type: "custom",
      customType: STATE_TYPE,
      data: { v: 1, modes: merged, opts },
      id: stateId,
      parentId: ctx.sessionManager.getLeafId(),
      timestamp: new Date().toISOString(),
    };

    const tmp = `${file}.tmp-${process.pid}`;
    try {
      writeFileSync(tmp, [...withoutState, stateEntry].map((e) => `${JSON.stringify(e)}\n`).join(""));
      renameSync(tmp, file);
    } catch (err) {
      ctx.ui.notify(`shake: failed to rewrite session file: ${err instanceof Error ? err.message : String(err)}`, "error");
      return;
    }

    const textTokens = estTokens(stats.toolChars + stats.bashChars + stats.blockChars + stats.thinkingChars);
    const kb = Math.round((stats.imageBytes * 0.75) / 1024);
    const parts: string[] = [`~${fmt(textTokens)} tokens freed`];
    if (stats.imageCount > 0) parts.push(`${stats.imageCount} image(s) removed (~${fmt(kb)} KB)`);
    // pi's footer anchors context usage to the last provider-reported call,
    // so the freed space shows up there after the next LLM call.
    const doneText = `shook history: ${parts.join(" · ")} — footer usage refreshes on the next LLM call`;

    // Re-read the rebuilt file into the live session manager (public API;
    // the ReadonlySessionManager type on ctx merely hides it). No session
    // switch: the running process keeps its state, and the context hook
    // keeps the provider payload shaken until the session is reloaded.
    try {
      const liveManager = ctx.sessionManager as unknown as { setSessionFile(path: string): void };
      liveManager.setSessionFile(file);
    } catch (err) {
      ctx.ui.notify(`shake: history rebuilt on disk, in-memory re-read failed: ${err instanceof Error ? err.message : String(err)}`, "error");
      return;
    }

    modes = merged;
    ctx.ui.notify(doneText, "info");
  };
  const showStatus = (ctx: ExtensionCommandContext): void => {
    const model = ctx.model as unknown as ModelLike;
    const stats = estimateRemovable(ctx, { tools: true, images: true, thinking: true });
    const usage = ctx.getContextUsage();
    const lines: string[] = [
      `pi-shake: tools=${modes.tools ? "ON" : "off"}  images=${modes.images ? "ON" : "off"}  thinking=${modes.thinking ? "ON" : "off"}`,
    ];
    if (usage) {
      const tok = usage.tokens != null ? fmt(usage.tokens) : "?";
      lines.push(`context: ${tok} / ${fmt(usage.contextWindow)} tokens${usage.percent != null ? ` (${usage.percent.toFixed(1)}%)` : ""}`);
    }
    const toolTok = estTokens(stats.toolChars + stats.bashChars);
    const blockTok = estTokens(stats.blockChars);
    const thinkTok = estTokens(stats.thinkingChars);
    const kb = Math.round((stats.imageBytes * 0.75) / 1024);
    lines.push(
      `removable now: ~${fmt(toolTok)} tokens tool/bash (${stats.toolCount}) · ~${fmt(blockTok)} tokens long text (${stats.blockCount})` +
        ` · ${stats.imageCount} images (~${fmt(kb)} KB) · ~${fmt(thinkTok)} tokens thinking (${stats.thinkingCount})`,
    );
    if (model?.api === "anthropic-messages" && !canDropSignedThinking(model)) {
      lines.push("note: signed thinking kept — this Anthropic model requires thinking blocks in replayed history");
    }
    lines.push("usage: /shake tools|images|thinking|all — rebuilds the session file in place");
    if (ctx.hasUI) {
      ctx.ui.setWidget("pi-shake", lines);
      ctx.ui.notify(lines[0] ?? "", "info");
    } else {
      ctx.ui.notify(lines.join(" | "), "info");
    }
  };

  pi.registerCommand("shake", {
    description: "Rebuild session history in place without heavy content (tool output, images, thinking)",
    getArgumentCompletions: (prefix) => {
      const items = KINDS.map((v) => ({ value: v, label: v }));
      const filtered = items.filter((i) => i.value.startsWith(prefix));
      return filtered.length > 0 ? filtered : null;
    },
    handler: async (args, ctx) => {
      const kind = (args ?? "").trim().toLowerCase();
      if (kind === "") {
        showStatus(ctx);
        return;
      }
      if ((KINDS as readonly string[]).includes(kind)) {
        const wanted: ShakeModes = {
          tools: kind === "tools" || kind === "all",
          images: kind === "images" || kind === "all",
          thinking: kind === "thinking" || kind === "all",
        };
        await rebuildHistory(ctx, wanted);
        return;
      }
      ctx.ui.notify(`unknown shake mode "${kind}" — try: tools, images, thinking, all`, "error");
    },
  });
}
