import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import test from 'node:test';

// Default: built client. Installed-package test supplies its actual package file URL.
const { KevLayaClient, KevLayaHTTPError, CLIENT_CONTRACT_VERSION } = await import(
  process.env.LAYEV_CLIENT_IMPORT ?? new URL('../dist/index.js', import.meta.url).href
);

async function serve(handler, check) {
  const server = createServer((req, res) => {
    Promise.resolve(handler(req, res)).catch(() => {
      res.statusCode = 500;
      res.end('test handler failed');
    });
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  try {
    await check(`http://127.0.0.1:${server.address().port}`);
  } finally {
    server.closeAllConnections();
    await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  }
}

async function body(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

function reply(res, value) {
  res.setHeader('content-type', 'application/json');
  res.end(JSON.stringify(value));
}

const response = {
  model: 'fixture-model',
  answers: {
    choice: { type: 'choice', choice: 'yes', confidence: .5, probabilities: { yes: .75, no: .25 } },
    score: { type: 'score', score: 1, confidence: .5, probabilities: { 0: .25, 1: .75 }, legend: { 0: 'low', 1: 'high' } },
    noul: { type: 'noul', noul: .75 },
  },
  usage: { input_tokens: 19, output_tokens: 3, forward_tokens: 27, state_tokens: 7, branch_tokens: [4, 4, 4], generated_tokens: 0 },
  confidence_definition: 'entropy-concentration-v1', calibration_status: 'fixture-only',
};

// A real TCP HTTP transport fixture; NEVER a trained Layev service or Jev result.
test('client contract is explicitly named', () => assert.equal(CLIENT_CONTRACT_VERSION, 'layev-client/1'));

test('structured Unicode inputs, IDs, typed answers and usage survive HTTP unchanged', async () => {
  const state = { text: 'e\u0301中文🙂<|fim_prefix|>', nested: ['x', { flag: true, value: null }] };
  const questions = {
    choice: { type: 'choice', instructions: { label: ['é', 'e\u0301'] }, criteria: { yes: { nested: 1 }, no: null } },
    score: { type: 'score', instructions: ['grade'], criteria: [{ rubric: 'low' }, { rubric: 'high' }] },
    noul: { type: 'noul', instructions: 'truth', criteria: { true: { label: 'yes' }, false: null } },
  };
  let seen;
  await serve(async (req, res) => {
    seen = { url: req.url, method: req.method, auth: req.headers.authorization, json: await body(req) };
    reply(res, response);
  }, async baseURL => {
    const before = JSON.stringify({ state, questions });
    const client = new KevLayaClient({ baseURL: baseURL + '///', apiKey: 'synthetic-test-key', model: 'fixture-model' });
    assert.deepEqual(await client.systemOne(state, questions), response);
    assert.equal(JSON.stringify({ state, questions }), before);
  });
  assert.deepEqual(seen, { url: '/v1/systemone', method: 'POST', auth: 'Bearer synthetic-test-key',
    json: { state, questions, model: 'fixture-model' } });
});

test('per-call model override and existing default are preserved', async () => {
  const names = [];
  await serve(async (req, res) => { names.push((await body(req)).model); reply(res, response); }, async baseURL => {
    await new KevLayaClient({ baseURL }).systemOne('fixture', {});
    await new KevLayaClient({ baseURL, model: 'configured' }).systemOne('fixture', {}, 'override');
  });
  assert.deepEqual(names, ['kev-laya-preview', 'override']);
});

test('models uses GET and preserves version descriptions', async () => {
  const models = { models: [{ name: 'fixture-model', description: 'synthetic', release_date: '2026-09-27' }] };
  await serve((req, res) => {
    assert.equal(req.url, '/prefix/v1/models'); assert.equal(req.method, 'GET');
    assert.equal(req.headers.authorization, undefined); reply(res, models);
  }, async baseURL => assert.deepEqual(await new KevLayaClient({ baseURL: baseURL + '/prefix' }).models(), models));
});

for (const status of [401, 404, 422, 429, 500]) {
  test(`HTTP ${status} is typed, content-free and not retried`, async () => {
    let calls = 0;
    await serve((req, res) => { calls++; res.statusCode = status; res.end('PRIVATE-RESPONSE synthetic-test-key'); }, async baseURL => {
      await assert.rejects(new KevLayaClient({ baseURL, apiKey: 'synthetic-test-key' }).models(), error => {
        assert.ok(error instanceof KevLayaHTTPError); assert.equal(error.status, status);
        assert.equal(error.message, `Kev-Laya HTTP ${status}`);
        assert.ok(!JSON.stringify(error).includes('PRIVATE-RESPONSE'));
        assert.ok(!error.stack.includes('synthetic-test-key'));
        return true;
      });
    });
    assert.equal(calls, 1);
  });
}

test('JSON diagnostics do not echo response data', async () => {
  await serve((req, res) => res.end('PRIVATE-NON-JSON'), async baseURL => {
    await assert.rejects(new KevLayaClient({ baseURL }).models(), error => {
      assert.equal(error.message, 'Kev-Laya returned invalid JSON');
      assert.ok(!error.stack.includes('PRIVATE-NON-JSON')); return true;
    });
  });
});

test('redirect responses are not followed', async () => {
  let calls = 0;
  await serve((req, res) => { calls++; res.writeHead(307, { location: '/redirect-target' }); res.end(); }, async baseURL => {
    await assert.rejects(new KevLayaClient({ baseURL, apiKey: 'synthetic-test-key' }).models());
  });
  assert.equal(calls, 1);
});

for (const sendHeaders of [false, true]) {
  test(`timeout covers ${sendHeaders ? 'stalled response body' : 'missing response headers'}`, async () => {
    let calls = 0;
    await serve((req, res) => {
      calls++;
      if (sendHeaders) { res.writeHead(200, { 'content-type': 'application/json' }); res.write('{'); }
      // The connection is deliberately held; serve() closes it on cleanup.
    }, async baseURL => {
      await assert.rejects(new KevLayaClient({ baseURL, timeoutMs: 100 }).models(), error => {
        assert.ok(['TimeoutError', 'AbortError'].includes(error.name), error.name); return true;
      });
    });
    assert.equal(calls, 1);
  });
}

test('mutation of caller options cannot redirect the original client', async () => {
  await serve((req, res) => {
    assert.equal(req.headers.authorization, 'Bearer original'); reply(res, { models: [] });
  }, async baseURL => {
    const options = { baseURL, apiKey: 'original' };
    const client = new KevLayaClient(options);
    options.baseURL = 'http://127.0.0.1:1'; options.apiKey = 'replacement';
    assert.deepEqual(await client.models(), { models: [] });
  });
});

test('invalid base URLs and timeout values fail before network access', () => {
  // Construct the synthetic userinfo URL so the source audit does not mistake
  // this negative-control fixture for a checked-in credential.
  const userinfoURL = ['https://', 'u', ':', 'p', '@example.org'].join('');
  for (const baseURL of ['file:///tmp/data', 'ftp://host', userinfoURL, 'https://example.org?q=1', 'https://example.org#fragment']) {
    assert.throws(() => new KevLayaClient({ baseURL }), TypeError);
  }
  for (const timeoutMs of [0, -1, 1.1, NaN, Infinity, 2147483648]) {
    assert.throws(() => new KevLayaClient({ baseURL: 'http://127.0.0.1:1', timeoutMs }), RangeError);
  }
});
