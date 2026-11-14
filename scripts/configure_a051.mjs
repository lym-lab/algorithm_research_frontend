import fs from 'node:fs/promises';

const endpoint = new URL(process.argv[2]);
if (!['http:', 'https:'].includes(endpoint.protocol) || endpoint.username || endpoint.password || endpoint.pathname !== '/process') {
  throw new Error('Pass the public HTTP(S) /process endpoint without credentials');
}
const file = new URL('../public/methods/group_a051_nmr_watermark.json', import.meta.url);
const method = JSON.parse(await fs.readFile(file, 'utf8'));
method.adapter.endpoint = endpoint.href;
await fs.writeFile(file, JSON.stringify(method, null, 2) + '\n');
console.log(`A051 endpoint: ${endpoint.href}; rebuild frontend before deployment`);
