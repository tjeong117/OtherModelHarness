import { execFile } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import { runPiEditor } from "./adapter.mjs";
import { workerEnvironment } from "./runtime.mjs";

const model = { id: "scripted-tiny-demo", name: "Scripted tiny demo", provider: "offline-demo",
  api: "openai-completions", reasoning: false, input: ["text"], contextWindow: 32768, maxTokens: 256,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
const cases = [{ slice: "improve", prompt: "color dax", answers: ["yes"] },
  { slice: "preserve", prompt: "color red", answers: ["yes"] },
  { slice: "preserve", prompt: "color blue", answers: ["no"] }];

// Offline policy for this fixture only. All observations and the patch come from the Python worker.
function nextAction(messages) {
  const previous = messages.findLast((m) => m.role === "toolResult");
  if (!previous) return ["inspect_target", { prompts: ["color red"], probe: { kind: "sae", layer: 0, positions: [-1] },
    intervention: { feature: 0, scale: 0 }, token_ids: [6, 7] }];
  const observation = JSON.parse(previous.content[0].text);
  switch (previous.toolName) {
    case "inspect_target": return ["test_behavior", {}];
    case "test_behavior": return ["read_weights", { tensor: "embed.weight", selection: [3], mode: "ref" }];
    case "read_weights": return ["preview_weights", { writes: [{ tensor: "embed.weight", selection: [5],
      operation: "copy", source: { handle: observation.handle } }] }];
    case "preview_weights":
      if (!observation.behavior.improve.fixed.length || observation.behavior.preserve.regressed.length) {
        throw new Error("The candidate missed the public improvement or preservation check");
      }
      return ["commit_preview", { candidate_id: observation.candidate_id }];
    case "commit_preview": return ["submit_patch", {}];
    default: throw new Error(`Unexpected demo tool result: ${previous.toolName}`);
  }
}

function demoStream(editor, context) {
  const [name, args] = nextAction(context.messages);
  const toolCall = { type: "toolCall", id: "demo-" + name, name, arguments: args };
  const message = { role: "assistant", api: editor.api, provider: editor.provider, model: editor.id,
    timestamp: Date.now(), stopReason: "toolUse", content: [toolCall],
    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
  const stream = createAssistantMessageEventStream();
  stream.push({ type: "start", partial: { ...message, content: [] } });
  stream.push({ type: "toolcall_start", contentIndex: 0, partial: message });
  stream.push({ type: "toolcall_end", contentIndex: 0, toolCall, partial: message });
  stream.push({ type: "done", reason: "toolUse", message });
  stream.end();
  return stream;
}

export async function runDemo({ out, python }) {
  const temporary = await mkdtemp(join(tmpdir(), "iwi-demo-"));
  try {
    const casesFile = join(temporary, "public-cases.json");
    await writeFile(casesFile, JSON.stringify(cases));
    const result = await runPiEditor({ model, streamFn: demoStream, out, python, maxTurns: 6,
      workerArgs: ["--target", "tiny", "--arm", "full", "--preload-sae", "--public-cases", casesFile],
      objective: "Inspect a live feature, teach dax the meaning of red, and preserve ordinary colors." });
    if (result.status !== "submitted") throw new Error(result.error || "Offline editor failed to submit its patch");
    const { stdout } = await promisify(execFile)(python, ["-P", "-m", "iwi_harness", "verify-tiny",
      "--patch", join(out, "patch.safetensors"), "--sha256", result.receipt.sha256], { env: workerEnvironment() });
    const verification = JSON.parse(stdout);
    await writeFile(join(out, "verification.json"), JSON.stringify(verification, null, 2) + "\n", { flag: "wx" });
    return { status: result.status, editor: result.editor, turns: result.turns, out, receipt: result.receipt, verification };
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}
