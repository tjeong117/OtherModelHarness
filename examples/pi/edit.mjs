// Preserve the original example command during migration to the packaged CLI.
import { main } from "../../agent/cli.mjs";
process.exitCode = await main(["edit", ...process.argv.slice(2)]);
