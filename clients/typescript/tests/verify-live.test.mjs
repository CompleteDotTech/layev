/** Real loopback TCP and the real CLI/client; every model reply is SYNTHETIC. */
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { once } from 'node:events';
import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const cli = fileURLToPath(new URL('../scripts/verify-live.mjs', import.meta.url));
const concrete = 'kev-laya-synthetic-test-version';
const alias = 'kev-laya-preview';
const publicRow = name => ({ name, description: 'Synthetic test only', release_date: '2026-09-27' });
const validModels = () => ({ models: [publicRow(concrete), publicRow(alias)], resolved_model: concrete });
const validResponse = () => ({
  model: concrete,
  answers: {
    choice: { type: 'choice', choice: 'red', confidence: .5, probabilities: { red: .8, blue: .2 } },
    score: { type: 'score', score: .75, confidence: .5,
      probabilities: { 0: .25, 1: .75 }, legend: { 0: 'level zero', 1: 'level one' } },
    noul: { type: 'noul', noul: .8 },
  },
  usage: { input_tokens: 19, output_tokens: 3, forward_tokens: 19,
    state_tokens: 7, branch_tokens: [4, 4, 4], generated_tokens: 0 },
  confidence_definition: 'entropy-concentration-v1', calibration_status: 'synthetic-unverified',
});

async function invoke(env) {
  const clean = { ...process.env };
  for (const key of ['LAYEV_BASE_URL', 'LAYEV_MODEL', 'LAYEV_API_KEY', 'LAYEV_CLIENT_IMPORT']) delete clean[key];
  return new Promise((resolve, reject) => execFile(process.execPath, [cli], {
    env: { ...clean, ...env }, timeout: 5000, maxBuffer: 1024 * 1024,
  }, (error, stdout, stderr) => {
    if (error && !Number.isInteger(error.code)) return reject(error);
    try { resolve({ code: error?.code ?? 0, receipt: JSON.parse(stdout), stdout, stderr }); }
    catch (parseError) { reject(new Error(`CLI did not produce JSON: ${parseError.message}`)); }
  }));
}

async function exchange({ models = validModels(), response = validResponse(), requested = concrete,
  status = 200, rawResponse } = {}) {
  const calls = [];
  const server = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    calls.push({ path: req.url, method: req.method, auth: req.headers.authorization,
      body: chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : null });
    res.setHeader('content-type', 'application/json');
    res.statusCode = status;
    res.end(status !== 200 ? 'SYNTHETIC-PRIVATE-BODY' : req.url === '/v1/models' ? JSON.stringify(models)
      : rawResponse ?? JSON.stringify(response));
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  try {
    const result = await invoke({ LAYEV_BASE_URL: `http://127.0.0.1:${server.address().port}`,
      LAYEV_MODEL: requested, LAYEV_API_KEY: 'synthetic-gate-key' });
    return { ...result, calls };
  } finally {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
}

function assertNotNative(receipt) {
  for (const field of ['native_weights_verified', 'quality_measured', 'official_typesafe_sdk_tested', 'jev_compared']) {
    assert.equal(receipt[field], false, field);
  }
}

for (const requested of [concrete, alias]) {
  test(`real CLI accepts advertised ${requested === alias ? 'alias resolved by service' : 'concrete version'} without quality claims`, async () => {
    const result = await exchange({ requested });
    assert.equal(result.code, 0);
    assert.equal(result.receipt.status, 'passed');
    assert.equal(result.receipt.requests_attempted, 2);
    assert.equal(result.calls.length, 2);
    assert.equal(result.calls[1].body.model, requested);
    assert.equal(result.calls[1].body.state.text, 'color=red; level=1; case=99999');
    assert.equal(result.calls[1].body.state.unicode, 'e\u0301中文🙂');
    assert.equal(result.calls[1].auth, 'Bearer synthetic-gate-key');
    assertNotNative(result.receipt);
    assert.match(result.receipt.request_sha256, /^[a-f0-9]{64}$/);
    assert.match(result.receipt.response_sha256, /^[a-f0-9]{64}$/);
    assert.ok(!result.stdout.includes('synthetic-gate-key'));
  });
}

test('published minimal models contract still supports an exact concrete version', async () => {
  const result = await exchange({ models: { models: [publicRow(concrete)] } });
  assert.equal(result.code, 0);
  assertNotNative(result.receipt);
});

const mutations = {
  'presence-only fake envelope': r => { r.answers = { choice: { type: 'choice', probabilities: {} },
    score: { type: 'score', legend: {} }, noul: { type: 'noul' } }; r.usage = {}; r.confidence_definition = 'arbitrary'; },
  'missing choice': r => { delete r.answers.choice.choice; },
  'choice outside dynamic keys': r => { r.answers.choice.choice = 'absent'; },
  'missing question answer': r => { delete r.answers.noul; },
  'extra question answer': r => { r.answers.extra = { type: 'noul', noul: 1 }; },
  'probabilities array': r => { r.answers.choice.probabilities = [.8, .2]; },
  'empty probabilities': r => { r.answers.choice.probabilities = {}; },
  'unexpected probability key': r => { r.answers.choice.probabilities = { red: .8, green: .2 }; },
  'negative probability': r => { r.answers.choice.probabilities = { red: -.2, blue: 1.2 }; },
  'string probability': r => { r.answers.choice.probabilities.red = '.8'; },
  'unnormalized probabilities': r => { r.answers.choice.probabilities = { red: .2, blue: .2 }; },
  'confidence above one': r => { r.answers.choice.confidence = 2; },
  'missing score probabilities': r => { delete r.answers.score.probabilities; },
  'out of range score': r => { r.answers.score.score = 7; },
  'nonnumeric score': r => { r.answers.score.score = 'one'; },
  'incorrect score legend keys': r => { r.answers.score.legend = { 1: 'level zero', 2: 'level one' }; },
  'incorrect score legend values': r => { r.answers.score.legend[0] = 'wrong criterion'; },
  'missing noul value': r => { delete r.answers.noul.noul; },
  'out of range noul': r => { r.answers.noul.noul = -1; },
  'empty usage': r => { r.usage = {}; },
  'zero usage for nonempty probe': r => { r.usage.input_tokens = 0; },
  'negative usage': r => { r.usage.input_tokens = -1; },
  'fractional usage': r => { r.usage.output_tokens = .5; },
  'unsafe integer usage': r => { r.usage.forward_tokens = 2 ** 53; },
  'boolean usage': r => { r.usage.state_tokens = true; },
  'missing branch tokens': r => { delete r.usage.branch_tokens; },
  'wrong branch count': r => { r.usage.branch_tokens = [12]; },
  'negative branch count': r => { r.usage.branch_tokens = [-1, 4, 4]; },
  'zero branch count for nonempty probe': r => { r.usage.branch_tokens = [0, 4, 4]; },
  'generative token claim': r => { r.usage.generated_tokens = 1; },
  'unknown confidence definition': r => { r.confidence_definition = 'arbitrary-v999'; },
  'missing calibration status': r => { delete r.calibration_status; },
  'unadvertised response model': r => { r.model = 'unadvertised'; },
  'extra top-level wire field': r => { r.extra = 'unexpected'; },
};
for (const [name, mutate] of Object.entries(mutations)) {
  test(`real CLI rejects ${name}`, async () => {
    const response = validResponse(); mutate(response);
    const result = await exchange({ response });
    assert.equal(result.code, 1, JSON.stringify(result.receipt));
    assert.equal(result.receipt.status, 'failed');
    assert.equal(result.receipt.requests_attempted, 2);
    assert.equal(result.calls.length, 2);
    assertNotNative(result.receipt);
  });
}

test('overflow JSON numeric probability is rejected after real parsing', async () => {
  const rawResponse = JSON.stringify(validResponse()).replace('"red":0.8', '"red":1e999');
  const result = await exchange({ rawResponse });
  assert.equal(result.code, 1);
  assert.equal(result.receipt.status, 'failed');
});

for (const [name, models] of Object.entries({
  'missing metadata': { models: [{ name: concrete }] },
  'duplicate model IDs': { models: [publicRow(concrete), publicRow(concrete)] },
  'unadvertised resolved ID': { models: [publicRow(concrete)], resolved_model: 'different' },
  'nonstring resolved ID': { models: [publicRow(concrete)], resolved_model: true },
})) {
  test(`real CLI rejects ${name} before decision request`, async () => {
    const result = await exchange({ models });
    assert.equal(result.code, 1);
    assert.equal(result.receipt.status, 'failed');
    assert.equal(result.receipt.requests_attempted, 1);
    assert.equal(result.calls.length, 1);
  });
}

test('missing configuration is blocked and makes zero requests', async () => {
  const result = await invoke({});
  assert.equal(result.code, 2);
  assert.equal(result.receipt.status, 'blocked');
  assert.equal(result.receipt.requests_attempted, 0);
});

test('non-loopback destination is blocked and makes zero requests', async () => {
  const result = await invoke({ LAYEV_BASE_URL: 'https://example.invalid', LAYEV_MODEL: concrete });
  assert.equal(result.code, 2);
  assert.equal(result.receipt.reason, 'non_loopback_endpoint_refused');
  assert.equal(result.receipt.requests_attempted, 0);
});

test('actual HTTP authentication failure is not retried or echoed', async () => {
  const result = await exchange({ status: 401 });
  assert.equal(result.code, 1);
  assert.equal(result.calls.length, 1);
  assert.equal(result.receipt.reason, 'request_or_configuration_failure');
  assert.ok(!result.stdout.includes('SYNTHETIC-PRIVATE-BODY'));
  assert.ok(!result.stdout.includes('synthetic-gate-key'));
});
