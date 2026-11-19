import { parseArgs } from "node:util";
import { createModels } from "@earendil-works/pi-ai";
import { openaiProvider } from "@earendil-works/pi-ai/providers/openai";
import { anthropicProvider } from "@earendil-works/pi-ai/providers/anthropic";
import { runPiEditor } from "./adapter.mjs";

const argv = process.argv.slice(2);
const split = argv.indexOf("--");
const { values } = parseArgs({ args: split < 0 ? argv : argv.slice(0, split), options: {
  provider: { type: "string", default: "openai" }, model: { type: "string" },
  objective: { type: "string" }, out: { type: "string" }, turns: { type: "string", default: "12" },
  "max-output-tokens": { type: "string", default: "1024" },
}});
const providers = new Map([["openai", openaiProvider], ["anthropic", anthropicProvider]]);
const factory = providers.get(values.provider);
if (!factory) throw new Error("Choose openai or anthropic, or pass your provider streamFn to runPiEditor()");
const models = createModels();
models.setProvider(factory());
const model = models.getModel(values.provider, values.model);
if (!model) throw new Error("Set --model to a model ID in the pinned Pi provider catalog");
const result = await runPiEditor({ model, streamFn: models.streamSimple.bind(models),
  objective: values.objective, out: values.out, maxTurns: Number(values.turns),
  maxOutputTokens: Number(values["max-output-tokens"]),
  workerArgs: split < 0 ? [] : argv.slice(split + 1) });
console.log(JSON.stringify(result, null, 2));
if (result.status !== "submitted") process.exitCode = 1;
