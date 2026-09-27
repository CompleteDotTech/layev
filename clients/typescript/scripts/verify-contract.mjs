/** Read-only checks for the current Layev wire contract, NOT model acceptance. */
export const LIVE_GATE_VERSION = 'live-loopback-interface/2';
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const text = value => typeof value === 'string' && value.length > 0;
const unit = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
const count = value => Number.isSafeInteger(value) && value >= 0;
const sameKeys = (value, keys) => record(value) && Object.keys(value).length === keys.length
  && keys.every(key => Object.hasOwn(value, key));
const fail = code => { throw new Error(code); };

/** Follow only a server-advertised concrete ID; never rewrite the requested alias. */
export function resolveAdvertisedModel(models, requested) {
  const code = 'models_contract_shape_mismatch';
  if (!record(models) || !Array.isArray(models.models) || !models.models.length) fail(code);
  const names = new Set();
  for (const row of models.models) {
    if (!record(row) || !text(row.name) || typeof row.description !== 'string'
        || !text(row.release_date) || names.has(row.name)) fail(code);
    names.add(row.name);
  }
  if (!names.has(requested)) fail('requested_model_not_advertised');
  // ModelsResponse also permits the minimal public envelope without this Layev
  // extension. In that case only exact requested/response identity is accepted.
  if (!Object.hasOwn(models, 'resolved_model')) return requested;
  if (!text(models.resolved_model) || !names.has(models.resolved_model)) fail(code);
  return models.resolved_model;
}

function distribution(value, keys) {
  if (!sameKeys(value, keys) || !keys.every(key => unit(value[key]))) return false;
  const total = keys.reduce((sum, key) => sum + value[key], 0);
  // Current serving uses double-precision softmax. This is a wire-level
  // normalization tolerance, not a change to any native numerical oracle.
  return Math.abs(total - 1) <= 1e-6;
}

/** Validate this probe's strings/JSON shapes without scoring its known regression. */
export function verifyInterfaceResponse(response, questions, resolvedModel) {
  const code = 'client_contract_shape_mismatch';
  if (!sameKeys(response, ['model', 'answers', 'usage', 'confidence_definition', 'calibration_status'])
      || response.model !== resolvedModel || response.confidence_definition !== 'entropy-concentration-v1'
      || !text(response.calibration_status) || !record(questions)) fail(code);
  const ids = Object.keys(questions);
  if (!sameKeys(response.answers, ids)) fail(code);
  for (const id of ids) {
    const question = questions[id];
    const answer = response.answers[id];
    if (!record(question) || !record(answer) || answer.type !== question.type) fail(code);
    if (question.type === 'choice') {
      if (!record(question.criteria)) fail(code);
      const keys = Object.keys(question.criteria);
      if (!keys.length || !sameKeys(answer, ['type', 'choice', 'confidence', 'probabilities'])
          || typeof answer.choice !== 'string' || !keys.includes(answer.choice)
          || !unit(answer.confidence) || !distribution(answer.probabilities, keys)) fail(code);
    } else if (question.type === 'score') {
      // The unchanged explicit probe uses string levels. Do not invent a JSON
      // rendering rule that could differ from Python canonical/render semantics.
      if (!Array.isArray(question.criteria) || question.criteria.length < 2
          || !question.criteria.every(level => typeof level === 'string')) fail(code);
      const keys = question.criteria.map((_, i) => String(i));
      if (!sameKeys(answer, ['type', 'score', 'confidence', 'probabilities', 'legend'])
          || typeof answer.score !== 'number' || !Number.isFinite(answer.score)
          || answer.score < 0 || answer.score > keys.length - 1 || !unit(answer.confidence)
          || !distribution(answer.probabilities, keys) || !sameKeys(answer.legend, keys)
          || !keys.every((key, i) => answer.legend[key] === question.criteria[i])) fail(code);
    } else if (question.type === 'noul') {
      if (!sameKeys(answer, ['type', 'noul']) || !unit(answer.noul)) fail(code);
    } else fail(code);
  }
  const usage = response.usage;
  if (!sameKeys(usage, ['input_tokens', 'output_tokens', 'forward_tokens', 'state_tokens',
    'branch_tokens', 'generated_tokens']) || !['input_tokens', 'output_tokens', 'forward_tokens',
      'state_tokens'].every(key => count(usage[key]) && usage[key] > 0) || usage.generated_tokens !== 0
      || !Array.isArray(usage.branch_tokens) || usage.branch_tokens.length !== ids.length
      || !usage.branch_tokens.every(value => count(value) && value > 0)) fail(code);
  // Values are reported, not independently tokenized/measured here. Likewise,
  // a calibration-status string is not proof of calibrated or trained weights.
}
