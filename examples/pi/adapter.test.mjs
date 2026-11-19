import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { streamSimple } from "@earendil-works/pi-ai/api/openai-completions";
import { runPiEditor } from "./adapter.mjs";

const python = process.env.IWI_PYTHON || "python3";
const fixture = [{ prompt: "color dax", answers: ["yes"], slice: "improve" }];
const shellTools = ["read", "write", "edit", "bash", "find", "ls", "grep"];

// A scripted local service, reached through Pi's real OpenAI-compatible streaming transport.
async function smoke(t, choose, workerArgs, maxTurns) {
  const requests = [];
  let serviceError;
  const server = createServer(async (req, res) => {
    try {
      const bytes = [];
      for await (const chunk of req) bytes.push(chunk);
      const body = JSON.parse(Buffer.concat(bytes));
      requests.push(body);
      const [name, args] = choose(body, requests.length);
      const tool = { index: 0, id: "call" + requests.length, type: "function",
        function: { name, arguments: JSON.stringify(args) } };
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      const chunk = (choices, usage) => res.write("data: " + JSON.stringify({
        id: "offline", object: "chat.completion.chunk", model: "offline", choices, usage,
      }) + "\n\n");
      chunk([{ index: 0, delta: { role: "assistant", tool_calls: [tool] }, finish_reason: null }]);
      chunk([{ index: 0, delta: {}, finish_reason: "tool_calls" }]);
      chunk([], { prompt_tokens: 20, completion_tokens: 2, total_tokens: 22 });
      res.end("data: [DONE]\n\n");
    } catch (err) {
      serviceError = err;
      res.writeHead(400);
      res.end("invalid test turn");
    }
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const out = await mkdtemp(join(tmpdir(), "iwi-pi-test-"));
  t.after(async () => { server.closeAllConnections(); server.close(); await rm(out, { recursive: true, force: true }); });
  const model = { id: "offline", name: "Offline fixture", api: "openai-completions", provider: "local-fixture",
    baseUrl: "http://127.0.0.1:" + server.address().port + "/v1", reasoning: false, input: ["text"],
    contextWindow: 32768, maxTokens: 384, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const result = await runPiEditor({ python, out, model, workerArgs, maxTurns, objective: "Teach dax the red answer.",
    streamFn: (editor, context, options) => streamSimple(editor, context, { ...options, apiKey: "offline" }) });
  if (serviceError) throw serviceError;
  return { out, requests, result, trace: JSON.parse(await readFile(join(out, "pi-trace.json"), "utf8")) };
}

test("real Pi loop and local transport preview, commit, submit and fresh-grade a tiny CPU patch", async (t) => {
  const { out, result, requests } = await smoke(t, (body, turn) => {
    const toolOutput = () => JSON.parse(body.messages.findLast((m) => m.role === "tool").content);
    switch (turn) {
      case 1: return ["test_behavior", { cases: fixture, max_new_tokens: 1 }];
      case 2: return ["read_weights", { tensor: "embed.weight", selection: [3], mode: "ref" }];
      case 3: return ["preview_weights", { cases: fixture, max_new_tokens: 1,
        writes: [{ tensor: "embed.weight", selection: [5], operation: "copy", source: { handle: toolOutput().handle } }] }];
      case 4: return ["commit_preview", { candidate_id: toolOutput().candidate_id }];
      case 5: return ["submit_patch", {}];
      default: throw new Error("unexpected extra editor inference after submission");
    }
  }, ["--arm", "C"], 6);
  const menu = requests[0].tools.map((t) => t.function.name);
  assert.ok(shellTools.every((tool) => !menu.includes(tool)));
  assert.ok(menu.includes("preview_weights"));
  assert.equal(requests[0].tools.find((t) => t.function.name === "test_behavior").function.parameters.properties.intervention, undefined);
  assert.equal(result.status, "submitted");
  assert.equal(result.turns, 5);
  assert.equal(result.usageByTurn[0].output, 2);
  const { stdout } = await promisify(execFile)(python,
    ["-m", "iwi_harness", "verify-tiny", "--patch", join(out, "patch.safetensors"), "--sha256", result.receipt.sha256]);
  assert.equal(JSON.parse(stdout).reward, 1);
});

test("Pi has only the selected interface and malformed/coerced input cannot mutate the worker", async (t) => {
  const { out, result, trace } = await smoke(t, (_body, turn) => turn === 1 ? ["bash", { command: "unused" }]
    : ["run_target", { prompts: ["color dax"], max_new_tokens: "2" }], ["--arm", "A"], 2);
  assert.equal(result.status, "no_submission");
  assert.equal(result.turns, 2);
  assert.equal(result.receipt, null);
  const responses = trace.messages.filter((m) => m.role === "toolResult");
  assert.equal(responses.length, 2);
  assert.ok(responses.every((m) => m.isError));
  const events = (await readFile(join(out, "trajectory.jsonl"), "utf8")).trim().split("\n").map(JSON.parse);
  assert.equal(events.length, 1);
  assert.equal(events[0].arguments.max_new_tokens, "2");
  assert.equal(events[0].target_usage.forwards, 0);
});
