/** Pack and install in a fresh consumer, then test the installed files and declarations. */
import { execFileSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const npmCli = process.env.npm_execpath;
if (!npmCli) throw new Error('Run this gate with npm run test:installed');
const require = createRequire(join(root, 'package.json'));
const compiler = require.resolve('typescript/bin/tsc');
assert.equal(require('typescript/package.json').version, '5.8.3', 'Use the pinned TypeScript compiler');
function npm(args, cwd, capture = false) {
  return execFileSync(process.execPath, [npmCli, ...args], {
    cwd, encoding: 'utf8', stdio: capture ? ['ignore', 'pipe', 'inherit'] : 'inherit',
    env: { ...process.env, npm_config_audit: 'false', npm_config_fund: 'false' },
  });
}
const temp = mkdtempSync(join(tmpdir(), 'layev-client-package-'));
try {
  npm(['run', 'build'], root);
  const [packed] = JSON.parse(npm(['pack', '--offline', '--ignore-scripts', '--json', '--pack-destination', temp], root, true));
  const expected = ['LICENSE', 'NOTICE', 'README.md', 'dist/index.d.ts', 'dist/index.js', 'package.json'];
  assert.deepEqual(packed.files.map(file => file.path).sort(), expected.sort());
  const consumer = join(temp, 'consumer'); mkdirSync(consumer);
  writeFileSync(join(consumer, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
  npm(['install', '--offline', '--ignore-scripts', '--no-audit', '--no-fund', join(temp, packed.filename)], consumer);
  const installed = join(consumer, 'node_modules', 'layev-client');
  assert.equal(readFileSync(join(installed, 'dist/index.js'), 'utf8'), readFileSync(join(root, 'dist/index.js'), 'utf8'));
  const typeSource = `import { KevLayaClient, type Question, type SystemOneResponse } from 'layev-client';
const client = new KevLayaClient({baseURL: 'http://127.0.0.1:1'});
const questions: Record<string, Question> = {
  choice: {type: 'choice', instructions: {nested: ['x']}, criteria: {a: null}},
  score: {type: 'score', instructions: null, criteria: ['low', 'high']},
  noul: {type: 'noul', instructions: ['true?'], criteria: {true: null}}
};
const answer: Promise<SystemOneResponse> = client.systemOne({text: 'x'}, questions);
// @ts-expect-error State cannot be null.
client.systemOne(null, questions);
// @ts-expect-error Choice criteria must be a label mapping, not a number.
const bad: Question = {type: 'choice', instructions: null, criteria: 1};
void answer; void bad;
`;
  writeFileSync(join(consumer, 'check.mts'), typeSource);
  execFileSync(process.execPath, [compiler, '--noEmit', '--strict', '--module', 'NodeNext', '--moduleResolution', 'NodeNext', '--target', 'ES2022', '--lib', 'ES2022,DOM', 'check.mts'], { cwd: consumer, stdio: 'inherit' });
  writeFileSync(join(consumer, 'import.mjs'), "import {KevLayaClient} from 'layev-client'; if (typeof KevLayaClient !== 'function') throw new Error('bad export');\n");
  execFileSync(process.execPath, ['import.mjs'], { cwd: consumer, stdio: 'inherit' });
  execFileSync(process.execPath, ['--test', join(root, 'tests/transport.test.mjs')], {
    cwd: consumer, stdio: 'inherit', env: { ...process.env, LAYEV_CLIENT_IMPORT: pathToFileURL(join(installed, 'dist/index.js')).href },
  });
  console.log(JSON.stringify({ gate: 'installed-typescript-transport-fixture', status: 'passed',
    trained_service: false, official_typesafe_sdk: false, jev_comparison: false,
    package: packed.filename, sha1: packed.shasum, integrity: packed.integrity }));
} finally {
  rmSync(temp, { recursive: true, force: true });
}
