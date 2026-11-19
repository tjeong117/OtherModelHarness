#!/usr/bin/env node
import { parseArgs } from "node:util";
import { createModels } from "@earendil-works/pi-ai";
import { openaiProvider } from "@earendil-works/pi-ai/providers/openai";
import { anthropicProvider } from "@earendil-works/pi-ai/providers/anthropic";
import { streamSimple } from "@earendil-works/pi-ai/api/openai-completions";
import packageInfo from "../package.json" with { type: "json" };
import { runPiEditor } from "./adapter.mjs";
import { setupWorker, workerPython } from "./runtime.mjs";

const text = { type: "string" };
const flag = { type: "boolean" };
const helpOption = { type: "boolean", short: "h" };
const workerOptions = {
  target: text, "target-revision": text, "target-blocks": text, device: text,
  "local-files-only": flag, "preload-sae": flag, arm: text, "public-cases": text,
  "max-tool-calls": text, "max-forwards": text, "max-forward-tokens": text,
};
const editorOptions = {
  provider: text, model: text, objective: text, out: text, turns: text,
  "max-output-tokens": text, "base-url": text, "context-window": text, help: helpOption,
};

const generalHelp = `Usage: iwi <command> [options]

  setup  Install the Python target worker
  demo   Run a complete offline Pi editing episode and verify its patch
  edit   Run a model editor with the target's weight and inspection tools

Use iwi <command> --help for options, or iwi --version for the installed version.`;

const help = {
  setup: "Usage: iwi setup [--hf] [--python EXECUTABLE]\n\n--hf installs Transformers target support. --python selects the interpreter for setup.",
  demo: "Usage: iwi demo --out DIR\n\nA scripted Pi editor edits a tiny CPU target. A fresh worker verifies the patch; results are written to DIR.",
  edit: `Usage: iwi edit --model ID --objective TEXT --out DIR [options]

  --provider openai|anthropic    Hosted Pi provider (default openai)
  --base-url URL                 OpenAI-compatible editor endpoint, including its API prefix
  --context-window N             Custom endpoint context window in tokens (default 32768)
  --turns N                      Maximum model turns (default 12)
  --max-output-tokens N          Maximum generated tokens per turn (default 1024)
  --target tiny|gemma|ORG/REPO   Editable target (default tiny)
  --target-revision REV          Checkpoint revision for a custom target
  --target-blocks PATH           Decoder block path for a custom target
  --device DEVICE                Target device, e.g. cpu or cuda (default cpu)
  --arm A|B|C|D|full             Worker tool menu (default full)
  --public-cases FILE            JSON array of public improvement/preservation cases
  --preload-sae                  Load a registered dictionary for the target
  --local-files-only             Use cached target artifacts
  --max-tool-calls N             Maximum worker calls (default 128)
  --max-forwards N               Maximum target forwards (default 256)
  --max-forward-tokens N         Maximum processed target tokens (default 262144)

Use a fresh output directory. Hosted editors use OPENAI_API_KEY or ANTHROPIC_API_KEY.
Custom endpoints use UPDATER_API_KEY when set. Worker options also work after --.`,
};

function required(values, name) {
  if (!values[name]?.trim()) throw new Error(`--${name} is required`);
  return values[name];
}

function positiveInteger(value, name) {
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value)) || Number(value) < 1) {
    throw new Error(`--${name} must be a positive integer`);
  }
  return Number(value);
}

function parseEdit(argv) {
  const split = argv.indexOf("--");
  const front = parseArgs({ args: split < 0 ? argv : argv.slice(0, split),
    options: { ...workerOptions, ...editorOptions } }).values;
  const tail = parseArgs({ args: split < 0 ? [] : argv.slice(split + 1), options: workerOptions }).values;
  for (const name of Object.keys(tail)) {
    if (name in front) throw new Error(`--${name} was set on both sides of --`);
  }
  return { ...front, ...tail };
}

function modelFor(values, maxOutputTokens) {
  const id = required(values, "model");
  if ("base-url" in values) {
    if (values.provider && !["local", "openai"].includes(values.provider)) {
      throw new Error("--base-url uses OpenAI-compatible transport; choose --provider local or openai");
    }
    const url = URL.parse(values["base-url"]);
    if (!url || !["http:", "https:"].includes(url.protocol) || url.username || url.password || url.hash || url.search) {
      throw new Error("--base-url must be an HTTP(S) API base URL without credentials, query or fragment");
    }
    const contextWindow = positiveInteger(values["context-window"] ?? "32768", "context-window");
    if (maxOutputTokens >= contextWindow) throw new Error("--max-output-tokens must be smaller than --context-window");
    const model = { id, name: id, provider: "local", api: "openai-completions", baseUrl: url.href.replace(/\/$/, ""),
      input: ["text"], contextWindow, maxTokens: maxOutputTokens, reasoning: false,
      compat: { supportsDeveloperRole: false, supportsStore: false, maxTokensField: "max_tokens" },
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
    return { model, streamFn: (m, ctx, options) => streamSimple(m, ctx,
      { ...options, apiKey: process.env.UPDATER_API_KEY || "local" }) };
  }
  if ("context-window" in values) throw new Error("--context-window requires --base-url");
  const provider = values.provider ?? "openai";
  const factories = new Map([["openai", openaiProvider], ["anthropic", anthropicProvider]]);
  if (!factories.has(provider)) throw new Error("Choose --provider openai or anthropic, or use --base-url for a custom endpoint");
  const registry = createModels();
  registry.setProvider(factories.get(provider)());
  const model = registry.getModel(provider, id);
  if (!model) throw new Error(`Unknown model ${id} in the pinned Pi ${provider} catalog; check --provider and --model`);
  return { model, streamFn: registry.streamSimple.bind(registry) };
}

async function edit(values) {
  const objective = required(values, "objective");
  const out = required(values, "out");
  const maxTurns = positiveInteger(values.turns ?? "12", "turns");
  const maxOutputTokens = positiveInteger(values["max-output-tokens"] ?? "1024", "max-output-tokens");
  const editor = modelFor(values, maxOutputTokens);
  const workerArgs = [];
  for (const [name, definition] of Object.entries(workerOptions)) {
    if (!(name in values)) continue;
    workerArgs.push("--" + name);
    if (definition.type === "string") {
      const value = required(values, name);
      if (name.startsWith("max-")) positiveInteger(value, name);
      workerArgs.push(value);
    }
  }
  return runPiEditor({ ...editor, python: await workerPython(), objective, out, maxTurns, maxOutputTokens, workerArgs });
}

export async function main(argv = process.argv.slice(2)) {
  try {
    const [command, ...args] = argv;
    if (command === "--version" || command === "-v") {
      console.log(packageInfo.version);
      return 0;
    }
    if (!command || ["-h", "--help", "help"].includes(command)) {
      console.log(generalHelp);
      return 0;
    }
    if (!Object.hasOwn(help, command)) throw new Error(`Unknown command "${command}". Use iwi --help`);
    const values = command === "edit" ? parseEdit(args)
      : parseArgs({ args, options: command === "setup" ? { help: helpOption, hf: flag, python: text }
                                                      : { help: helpOption, out: text } }).values;
    if (values.help) {
      console.log(help[command]);
      return 0;
    }
    if (command === "setup") {
      const python = await setupWorker({ hf: values.hf ?? false, python: values.python });
      console.log(`Worker ready at ${python}`);
      return 0;
    }
    const result = command === "edit" ? await edit(values)
      : await (await import("./demo.mjs")).runDemo({ out: required(values, "out"), python: await workerPython() });
    console.log(JSON.stringify(result, null, 2));
    return result.status === "submitted" && (command !== "demo" || result.verification.reward === 1) ? 0 : 1;
  } catch (error) {
    console.error(`iwi: ${error.message}`);
    return 1;
  }
}

if (import.meta.main) process.exitCode = await main();
