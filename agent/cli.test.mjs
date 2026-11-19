import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const cli = fileURLToPath(new URL("./cli.mjs", import.meta.url));
const exec = promisify(execFile);
const invoke = (args, options) => exec(process.execPath, [cli, ...args], options);

test("CLI presents help and reports invalid options before starting an episode", async () => {
  assert.match((await invoke(["edit", "--help"])).stdout, /--base-url/);
  await assert.rejects(invoke(["edit", "--model", "unknown", "--out", "unused"]), /--objective is required/);
  await assert.rejects(invoke(["edit", "--model", "m", "--objective", "test", "--out", "unused", "--turns", "0"]),
    /--turns must be a positive integer/);
  await assert.rejects(invoke(["edit", "--target", "tiny", "--", "--target", "gemma"]), /both sides/);
});

test("iwi demo replays the exported numerical patch and preserves both controls", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "iwi-cli-demo-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const out = join(dir, "episode");
  const result = JSON.parse((await invoke(["demo", "--out", out])).stdout);
  const verified = JSON.parse(await readFile(join(out, "verification.json")));
  assert.deepEqual(result.verification, verified);
  assert.equal(result.status, "submitted");
  assert.equal(result.turns, 6);
  assert.equal(verified.patch_sha256, result.receipt.sha256);
  assert.equal(verified.reward, 1);
  assert.equal(verified.retained, 2);
  const events = (await readFile(join(out, "trajectory.jsonl"), "utf8")).trim().split("\n").map(JSON.parse);
  assert.deepEqual(events.map((event) => event.tool),
    ["inspect_target", "test_behavior", "read_weights", "preview_weights", "commit_preview", "submit_patch"]);
  assert.deepEqual(events[1].arguments, {});
  assert.deepEqual(Object.keys(events[3].arguments), ["writes"]);
});

test("iwi edit streams from a custom provider with flat and legacy worker flags", async (t) => {
  const dir = await mkdtemp(join(tmpdir(), "iwi-cli-server-"));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const requests = [];
  const server = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    requests.push({ path: req.url, auth: req.headers.authorization, body: JSON.parse(Buffer.concat(chunks)) });
    const response = (delta, finish_reason) => "data: " + JSON.stringify({ id: "local",
      object: "chat.completion.chunk", model: "local-test-model", choices: [{ index: 0, delta, finish_reason }] }) + "\n\n";
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    res.end(response({ role: "assistant", tool_calls: [{ index: 0, id: "1", type: "function",
      function: { name: "submit_patch", arguments: "{}" } }] }, null) + response({}, "tool_calls") + "data: [DONE]\n\n");
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(() => { server.closeAllConnections(); server.close(); });
  const result = JSON.parse((await invoke(["edit", "--model", "local-test-model", "--base-url",
    `http://127.0.0.1:${server.address().port}/v1`, "--objective", "Submit the original weights.",
    "--out", join(dir, "episode"), "--target", "tiny", "--max-output-tokens", "96", "--", "--arm", "C"],
  { env: { ...process.env, UPDATER_API_KEY: "test-local-key", OPENAI_API_KEY: "different-test-key" } })).stdout);
  assert.equal(result.status, "submitted");
  assert.equal(result.receipt.changed_parameters, 0);
  assert.equal(requests.length, 1);
  const [{ path, auth, body }] = requests;
  assert.equal(path, "/v1/chat/completions");
  assert.equal(auth, "Bearer test-local-key");
  assert.equal(body.stream, true);
  assert.equal(body.model, "local-test-model");
  assert.equal(body.max_tokens, 96);
  assert.equal(body.messages[0].role, "system");
  const names = body.tools.map((t) => t.function.name);
  assert.ok(names.includes("preview_weights"));
  assert.ok(!names.includes("inspect_target"));
});
