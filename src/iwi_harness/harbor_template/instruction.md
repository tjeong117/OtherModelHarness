# Teach the target a color alias

Edit the offline transformer so that it answers prompts ending in `dax` as it would answer the corresponding prompt ending in `red`. Preserve its responses for familiar colors. You can practice on `color red`, `color blue`, and `color dax`. The verifier starts a fresh target, applies your numerical patch, and measures alias behavior and retention on unseen prompts.

Run `edit-target --schemas` to see the editor's tools. Run `edit-target` to start a single editing session. It accepts one JSON object per line on standard input and prints a JSON result after each action. Keep the same process open for observations and writes that share tensor handles. You can also pipe a file of actions to the command. For example, this public probe measures the last prompt token's residual at block 0:

```sh
echo '{"name":"inspect_target","arguments":{"prompts":["color red","color dax"],"probe":{"kind":"residual","layer":0}}}' | edit-target
```

Within the session that made the edit, send `{"name":"submit_patch","arguments":{}}` to save the patch. Submit `/logs/artifacts/patch.safetensors` as the result.

Use `edit-target --public-experiments --schemas` to inspect the optional paired behavior and weight-preview tools. Start the session with `edit-target --public-experiments` to enable them. The tiny target supports temporary feature scaling with its supplied SAE.

To reuse a set of public examples, pass `--public-cases FILE`. Each item in the JSON array contains a prompt, acceptable answers, and an `improve` or `preserve` slice. With this option, `test_behavior` accepts empty arguments and `preview_weights` needs only the proposed writes.
