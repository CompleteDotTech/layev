/** Explicit, fail-closed loopback interface gate. No implicit remote/paid requests. */
import { createHash } from 'node:crypto';
import { writeFileSync } from 'node:fs';
import { KevLayaClient, CLIENT_CONTRACT_VERSION } from '../dist/index.js';
import { LIVE_GATE_VERSION, resolveAdvertisedModel, verifyInterfaceResponse } from './verify-contract.mjs';

const baseURL = process.env.LAYEV_BASE_URL;
const model = process.env.LAYEV_MODEL;
const output = process.argv[2];
const receipt = { gate: 'live-loopback-interface', gate_version: LIVE_GATE_VERSION, contract: CLIENT_CONTRACT_VERSION,
  status: 'blocked', native_weights_verified: false, quality_measured: false,
  official_typesafe_sdk_tested: false, jev_compared: false, requests_attempted: 0 };
const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
try {
  if (!baseURL || !model) throw new Error('missing_explicit_base_url_or_model');
  const url = new URL(baseURL);
  if (!['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)) throw new Error('non_loopback_endpoint_refused');
  const client = new KevLayaClient({ baseURL, model, apiKey: process.env.LAYEV_API_KEY, timeoutMs: 30000 });
  receipt.status = 'failed';
  receipt.requests_attempted++;
  const models = await client.models();
  const resolvedModel = resolveAdvertisedModel(models, model);
  const state = { text: 'color=red; level=1; case=99999', unicode: 'e\u0301中文🙂' };
  const questions = {
    choice: { type: 'choice', instructions: 'Choose the color.', criteria: { red: 'red', blue: 'blue' } },
    score: { type: 'score', instructions: 'Rate the level.', criteria: ['level zero', 'level one'] },
    noul: { type: 'noul', instructions: 'The color is red.' },
  };
  receipt.requests_attempted++;
  const response = await client.systemOne(state, questions);
  verifyInterfaceResponse(response, questions, resolvedModel);
  Object.assign(receipt, { status: 'passed', model, requested_model: model, resolved_model: resolvedModel, request_sha256: hash({ state, questions, model }),
    response_sha256: hash(response), models_sha256: hash(models),
    note: 'Interface receipt only; does not score or repair the preserved quality regression.' });
} catch (error) {
  // Only these locally authored codes are safe to publish; network errors may contain URLs.
  const allowed = ['missing_explicit_base_url_or_model', 'non_loopback_endpoint_refused',
    'requested_model_not_advertised', 'models_contract_shape_mismatch', 'client_contract_shape_mismatch'];
  receipt.reason = allowed.includes(error.message) ? error.message : 'request_or_configuration_failure';
  process.exitCode = receipt.status === 'blocked' ? 2 : 1;
}
const json = JSON.stringify(receipt, null, 2) + '\n';
if (output) writeFileSync(output, json, { flag: 'wx' });
console.log(json);
