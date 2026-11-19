import { spawn } from "node:child_process";
import { access, link, mkdir, mkdtemp, readFile, rename, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { delimiter, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const { version } = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
const runtime = join(process.env.IWI_WORKER_HOME || process.env.XDG_DATA_HOME || join(homedir(), ".local", "share"),
  "other-model-harness", version);
const interpreter = (env) => process.platform === "win32" ? join(env, "Scripts", "python.exe") : join(env, "bin", "python");

export function workerEnvironment() {
  return { ...process.env, PYTHONPATH: [join(root, "src"), process.env.PYTHONPATH].filter(Boolean).join(delimiter) };
}

async function existing(path) {
  try { await access(path); return path; }
  catch (err) { if (err.code !== "ENOENT") throw err; }
}

async function installed(profile) {
  const record = join(runtime, profile + ".json");
  if (!await existing(record)) return;
  const env = JSON.parse(await readFile(record, "utf8")).environment;
  return existing(interpreter(env));
}

export async function workerPython() {
  if (process.env.IWI_PYTHON) return process.env.IWI_PYTHON;
  const managed = await installed("hf") || await installed("base");
  if (managed) return managed;
  const local = await existing(interpreter(join(root, ".venv")));
  if (local) return local;
  if (process.env.VIRTUAL_ENV) return interpreter(process.env.VIRTUAL_ENV);
  throw new Error("Run iwi setup to install the target worker, or set IWI_PYTHON to an environment with PyTorch and the worker dependencies.");
}

function run(program, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(program, args, { stdio: "inherit" });
    child.once("error", reject);
    child.once("close", (code) => code === 0 ? resolve() : reject(new Error(`${program} exited with code ${code}`)));
  });
}

/** Publish a worker environment only after installing and loading its dependencies. */
export async function setupWorker({ hf = false, python = "python3" } = {}) {
  const profile = hf ? "hf" : "base";
  const ready = await installed("hf") || (!hf && await installed("base"));
  if (ready) return ready;
  await mkdir(runtime, { recursive: true });
  const environment = await mkdtemp(join(runtime, profile + "-"));
  const exe = interpreter(environment);
  try {
    await run(python, ["-c", "import sys; sys.exit('Python 3.11+ is required') if sys.version_info < (3, 11) else None"]);
    await run(python, ["-m", "venv", environment]);
    await run(exe, ["-m", "pip", "install", hf ? root.replace(/[\\/]$/, "") + "[hf]" : root]);
    await run(exe, ["-I", "-c", hf ? "import iwi_harness, transformers, huggingface_hub" : "import iwi_harness"]);
    const pending = join(environment, "ready.json");
    await writeFile(pending, JSON.stringify({ environment, profile, version }) + "\n");
    const record = join(runtime, profile + ".json");
    try {
      await link(pending, record);
    } catch (err) {
      if (err.code !== "EEXIST") throw err;
      const existingWorker = await installed(profile);
      if (existingWorker) {
        await rm(environment, { recursive: true, force: true });
        return existingWorker;
      }
      // Replace a stale record whose interpreter was removed.
      await rename(pending, record);
    }
    return exe;
  } catch (err) {
    await rm(environment, { recursive: true, force: true });
    throw err;
  }
}
