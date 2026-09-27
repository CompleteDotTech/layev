# Layev TypeScript client

An independent, installable ESM HTTP client for Layev. Node 20 or later is required;
TypeScript 5.8.3 is pinned for development. The package is private to prevent
accidental registry publication; local `npm pack` and tarball installation work.
It is not the official TypeSafe SDK and does not establish Jev compatibility,
model quality, CUDA execution, or trained context capability.

## Build and install

From `clients/typescript` in a clean Layev checkout, with authorized npm access:

```sh
npm ci --ignore-scripts --no-audit --no-fund
npm run typecheck
npm test
npm run test:installed
npm pack
# In the consuming project:
npm install /absolute/path/to/layev-client-0.1.0-dev.1.tgz
```

There are no runtime package dependencies. The tarball contains only the compiled
JavaScript, declarations, package metadata, this document and license/notice.
`test:installed` makes a fresh temporary consumer, installs the packed artifact
offline, compiles a consumer against its installed declarations, imports the
package by name, and runs real loopback TCP transport tests using its installed
JavaScript. The temporary files are removed afterward. These transport tests use
synthetic server responses, not an installed Layev model service.
The existing `test.mjs` frontend-helper tests remain separate and unchanged.

```ts
import { KevLayaClient, KevLayaHTTPError } from 'layev-client';

const client = new KevLayaClient({
  baseURL: 'http://127.0.0.1:8000',
  model: 'kev-laya-preview',
  timeoutMs: 30_000,
  // Supply apiKey from the application's secret manager when required.
});
try {
  const result = await client.systemOne(
    { message: 'A structured state', tags: ['example'] },
    { relevant: { type: 'noul', instructions: 'The message concerns this task.' } },
  );
  console.log(result.model, result.answers.relevant.type);
} catch (error) {
  if (error instanceof KevLayaHTTPError) console.error(error.status);
  else throw error;
}
```

## Contract and security

`CLIENT_CONTRACT_VERSION` is `layev-client/1`. The existing JSON field names,
Choice/Score/Noul union, usage fields and default model name are preserved.
Type declarations describe the current Layev contract; they are **not runtime
JSON-schema validation**. Model identity is returned by the server, not inferred
from the package version. Choice and Score include distributions/confidence;
Score includes its legend; Noul remains its own numeric answer form.

Credentials appear only in the Authorization header. URL userinfo, query strings
and fragments are refused; options are copied so caller mutation cannot redirect
an existing client. Use HTTPS for non-loopback deployments. Redirects and automatic
retries are disabled. Timeouts cover headers and body reads. HTTP errors provide
a typed status but deliberately omit raw response bodies, which may echo private
data. Invalid-JSON errors also omit raw response content.

## Actual service gate (separate, never silently skipped)

After building, set `LAYEV_BASE_URL` to the existing authorized loopback service,
`LAYEV_MODEL` to its advertised installed model and, only when required,
`LAYEV_API_KEY` through the existing secret-safe route. Run:

```sh
npm run verify:live -- /path/to/new-interface-receipt.json
```

The command exits 2, not success, when explicit endpoint/model prerequisites are
absent. Only loopback addresses are accepted by this verification script. It checks
model advertisement and the mixed response contract using a fixed synthetic
request and writes content-free hashes. An existing receipt is never overwritten.
This is an interface gate, **not** an official TypeSafe SDK gate or a quality test:
it deliberately does not declare the `case=99999` regression fixed. It cannot
verify the service's checkpoint, native weights or CUDA use from HTTP shape alone.
The official SDK, installed Layev wheel, model/provenance and repository-wide
checks remain independent acceptance requirements in Layev issue #6.

## Development-lock provenance

The portable lock pins TypeScript 5.8.3 and its registry-published SHA-512 integrity
from https://registry.npmjs.org/typescript/5.8.3. It contains no workstation-local
package links. In the restricted execution session, compilation used the already
available TypeScript 5.8.3 toolchain; a fresh registry dependency install was not
possible. The installed **client tarball** test did run offline in a fresh consumer.
The existing `source-integrity-and-critical-lint` CI job now also runs the client
lock install, typecheck, source transport tests and installed-tarball test. Its name
and all four Python model-job names are unchanged. Hosted validation of this patch
remains required.
