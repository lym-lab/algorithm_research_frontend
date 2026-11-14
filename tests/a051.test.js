import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateManifest, validateParameters, defaultsFor } from '../src/lib/contract.js';

const method = JSON.parse(await readFile(new URL('../public/methods/group_a051_nmr_watermark.json', import.meta.url), 'utf8'));

test('A051正式方法符合平台契约，样例消息与真实音频文件完整', async () => {
  assert.equal(validateManifest(method), method);
  assert.deepEqual(validateParameters(method, defaultsFor(method)), {});
  assert.equal(method.examples[0].parameters.message.length, 256);
  assert.deepEqual(validateParameters(method, method.examples[0].parameters), {});
  const index = JSON.parse(await readFile(new URL('../public/methods/index.json', import.meta.url), 'utf8'));
  assert.equal(index.filter(name => name === `${method.id}.json`).length, 1);
  for (const item of ['input', 'output']) {
    const bytes = await readFile(new URL(`../public/${method.examples[0][item].src}`, import.meta.url));
    assert.equal(bytes.subarray(0, 4).toString(), 'RIFF');
    assert.ok(bytes.length > 1000);
  }
});
