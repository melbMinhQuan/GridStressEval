// Exercise the shipped HTML's event handlers without a browser dependency.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('apps/viewer.html', 'utf8');
const elements = new Map();
function element() {
  return {
    value: '', hidden: false, disabled: false,
    classList: {add() {}, remove() {}}, appendChild() {}, click() {},
    set innerHTML(value) {
      for (const match of value.matchAll(/id="([^"]+)"/g)) elements.set(match[1], element());
    },
  };
}
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  },
  createElement: element,
};
const defaults = {small:20, medium:40, large:100, customers:40, threshold:1000,
  probability_small:50, probability_medium:50, probability_large:50,
  type_small:33.4, type_medium:33.3, type_large:33.3};
const requests = [], timers = new Map();
let timerId = 0;
const context = vm.createContext({document, URL, URLSearchParams, Blob,
  setTimeout(fn) {timers.set(++timerId, fn); return timerId;},
  clearTimeout(id) {timers.delete(id);},
  async fetch(url) {
    if (url === '/api/config') return {ok:true, json:async()=>defaults};
    requests.push(new URL(url, 'http://localhost').searchParams);
    return {ok:true, json:async()=>({inputs:{method:'exact_binomial'},
      overall:{violation_probability:.123}, results:[], png:'test', source:'exact_binomial'})};
  },
});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
async function settle() { await new Promise(resolve=>setImmediate(resolve)); }
async function flush() {
  for (const [id, fn] of [...timers]) {timers.delete(id); fn();}
  await settle();
}
(async()=>{
  await settle();
  for (const key of ['probability_small', 'probability_medium', 'probability_large']) {
    const field = document.getElementById(key+'-number');
    field.value = '23.4'; field.onchange({target:field});
    await flush();
    assert.equal(requests.at(-1).get(key), '23.4');
    assert.equal(Number(document.getElementById(key).value), 23.4);
  }
  const slider = document.getElementById('probability_small');
  slider.value = '56.7'; slider.oninput({target:slider}); slider.onchange();
  await flush();
  assert.equal(requests.at(-1).get('probability_small'), '56.7');
  document.getElementById('reset').onclick();
  await flush();
  for (const [key, value] of Object.entries(defaults))
    assert.equal(Number(requests.at(-1).get(key)), value);
  assert.equal(document.getElementById('overall-value').textContent, '12.30%');
  assert.equal(document.getElementById('json').disabled, false);
  console.log('Local HTML decimal entry, slider synchronization, reset and display passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
