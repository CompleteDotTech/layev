# Official TypeSafe Python SDK interface gate

This is an opt-in interface check for Layev issue #6. It uses the public
`typesafe-sdk==0.7.2` package and its documented `TypeSafeClient`, `Choice`,
`Score`, and `Noul` types. It sends only to an explicit loopback Layev service.
The URL must be the origin (`http://127.0.0.1:<port>`), with no `/v1` suffix;
the SDK appends `/v1/systemone` itself. The SDK is not part of the repository's
locked runtime, so a separate consumer environment must install it explicitly.

Build the Layev wheel from the inspected source and install it into that clean
consumer environment. Start one owned authenticated loopback service from a
locally verified trained checkpoint. Set `KEV_LAYA_API_KEYS` for the server
to a JSON array containing a disposable local test key. Then, in the consumer
environment, run:

```sh
python scripts/verify_official_sdk.py self-test
KEV_LAYA_BASE_URL=http://127.0.0.1:<port> \
KEV_LAYA_MODEL=<concrete-model-id> \
KEV_LAYA_TEST_API_KEY=<local-test-key> \
python scripts/verify_official_sdk.py installed-service --out <new-private-report.json>
```

The installed-service gate requires a new receipt path, refuses non-loopback
URLs and unknown SDK versions, and checks a mixed structured state with
Choice/Score/Noul, dynamic choice criteria, typed answers, probabilities,
confidence, score legend, resolved model, nonzero input usage, and typed 401
and 422 errors. Neither the key nor response content is written to the receipt.
The self-test uses a synthetic in-process TCP server and is not native evidence.
The official SDK package and public contract are documented at
<https://docs.typesafe.ai/sdk/python>.

## Measured local result, 2026-09-27/28

The [raw receipt](evidence/sdk/official-sdk-trained-20260927.json) is from a
clean Python 3.13.12 consumer environment outside the checkout, with the
built Layev wheel SHA-256
`e04d768e1439cf1653a4dbf2f12679218b7d16faf070d1e07c2fb1092180fb84`
and official `typesafe-sdk==0.7.2` installed from public PyPI. The wheel's
27 runtime modules matched the source head of native PR #24, now merged at
`937fee343fdc758cd241e3e2bb48090a60ddd55f`. The service loaded the
actual two-step trained checkpoint SHA-256
`b6e0db32a65bbf240a5cdec8c938040054a45284e5d4cf35cd310160262b6efb`
on the RTX 3060. Its `/v1/models` response reported native weights loaded,
the concrete model ID in the receipt, and the tokenizer-v4 identity recorded
in the native training receipt. The server was stopped after the probe.

The real official SDK returned all three requested typed answers and token
usage. An incorrect local test key raised its typed authentication error;
an unknown model raised its typed unprocessable-entity error. The raw receipt
SHA-256 is `c7166acbcb08ecfee06681492d128abfa3ec69eebd7613dbbe2a147005f4b38d`;
its script SHA-256 matches the source file. The private trained checkpoint and
key are not in Git. This validates the tested interface, not representative
quality, calibrated confidence, proprietary TypeSafe internals, or Jev parity.
The TypeScript installed-client transport and timeout gates remain separate.
