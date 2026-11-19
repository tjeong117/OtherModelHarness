import { spawn } from "node:child_process";
import { once } from "node:events";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { Agent } from "@earendil-works/pi-agent-core";

const workerPath = fileURLToPath(new URL("./worker.py", import.meta.url));

/** Run one Pi Agent, one fresh Python EditingSession and one public objective. */
export async function runPiEditor({
  model, streamFn, objective, out, workerArgs = [], maxTurns = 12, maxOutputTokens = 1024,
  python = process.env.IWI_PYTHON || "python3",
}) {
  if (!Number.isSafeInteger(maxTurns) || maxTurns < 1) throw new Error("maxTurns must be a positive integer");
  if (!Number.isSafeInteger(maxOutputTokens) || maxOutputTokens < 1) throw new Error("maxOutputTokens must be a positive integer");
  if (!objective?.trim() || !out || !model || typeof streamFn !== "function") {
    throw new Error("Supply an objective, output directory, Pi model and streamFn");
  }
  const child = spawn(python, ["-u", workerPath, ...workerArgs, "--out", out], { stdio: ["pipe", "pipe", "inherit"] });
  let workerError;
  child.stdin.on("error", (err) => { workerError = err; });
  const exited = once(child, "close");
  // Capture spawn failures immediately; awaits below also observe this failure.
  exited.catch(() => {});
  const lines = createInterface({ input: child.stdout, crlfDelay: Infinity })[Symbol.asyncIterator]();
  async function receive() {
    const entry = await lines.next();
    if (entry.done) {
      const [code, signal] = await exited;
      throw new Error(`IWI worker exited ${signal || code} before replying`);
    }
    return JSON.parse(entry.value);
  }
  try {
    const { tools: menu, info } = await receive();
    let receipt = null;
    let turns = 0;
    const rawCalls = new Map();
    const tools = menu.map(({ function: schema }) => ({
      name: schema.name, label: schema.name, description: schema.description, parameters: schema.parameters,
      async execute(id, parsed) {
        // Pi validates before execution and may coerce values. The worker owns strict validation.
        const args = rawCalls.has(id) ? rawCalls.get(id) : parsed;
        const request = JSON.stringify({ name: schema.name, arguments: args }) + "\n";
        let response;
        try {
          if (workerError) throw workerError;
          await new Promise((resolve, reject) => child.stdin.write(request, (err) => err ? reject(err) : resolve()));
          response = await receive();
        } catch (err) {
          workerError = err;
          throw err;
        }
        if ("error" in response) throw new Error(response.error);
        if (schema.name === "submit_patch") receipt = response.result;
        return { content: [{ type: "text", text: JSON.stringify(response.result) }], details: null };
      },
    }));
    const agent = new Agent({
      initialState: { model, tools, systemPrompt: "Use the public target editing tools to complete the objective. " +
        `You have ${maxTurns} model turns. Finish by calling submit_patch. Session ${JSON.stringify(info)}` },
      toolExecution: "sequential",
      streamFn: (editor, context, options) => streamFn(editor, context, {
        ...options, maxTokens: Math.min(maxOutputTokens, editor.maxTokens),
      }),
      finishTurn() {
        turns++;
        if (receipt || workerError || turns >= maxTurns) return { action: "end" };
      },
    });
    agent.subscribe((event) => {
      if (event.type === "tool_execution_start") rawCalls.set(event.toolCallId, event.args);
      if (event.type === "tool_execution_end") rawCalls.delete(event.toolCallId);
    });
    await agent.prompt(objective);
    const messages = agent.state.messages;
    const replies = messages.filter((m) => m.role === "assistant");
    const last = replies.at(-1);
    let status = "no_submission";
    if (receipt) status = "submitted";
    else if (workerError) status = "worker_error";
    else if (["error", "aborted"].includes(last?.stopReason)) status = "editor_error";
    const result = { status, receipt, turns: replies.length, editor: { provider: model.provider, id: model.id },
      stopReason: last?.stopReason, error: workerError?.message || last?.errorMessage,
      maxTurns, maxOutputTokens, usageByTurn: replies.map((m) => m.usage) };
    await writeFile(join(out, "pi-trace.json"), JSON.stringify({ result, messages }, null, 2) + "\n", { flag: "wx" });
    return result;
  } finally {
    child.stdin.end();
    await exited;
  }
}
