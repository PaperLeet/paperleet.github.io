// The first advertised protocol. Older hosts have no protocol field and only
// support the original computed-output program contract.
// Keep the established wire identifier across the TorchWasm package rename.
export const PROTOCOL = Object.freeze({
  name: 'stridetorch-matrix-rpc', major: 1, minor: 0,
  features: Object.freeze(['program-matmul-f32', 'program-retain-inputs']),
});
const LEGACY = Object.freeze({
  name: PROTOCOL.name, major: 0, minor: 0, legacy: true,
  features: Object.freeze(['program-matmul-f32']),
});
const invalid = message => Object.assign(new Error(message), {code: 'IncompatibleProtocol'});

export function negotiateProtocol(ready) {
  if (!Object.hasOwn(ready, 'protocol')) return LEGACY;
  const value = ready.protocol;
  if (!value || Object.getPrototypeOf(value) !== Object.prototype ||
      value.name !== PROTOCOL.name || value.major !== PROTOCOL.major ||
      !Number.isSafeInteger(value.minor) || value.minor < 0 ||
      !Array.isArray(value.features) || value.features.length > 32 ||
      value.features.some(feature => typeof feature !== 'string' || !/^[a-z0-9-]{1,64}$/.test(feature)) ||
      new Set(value.features).size !== value.features.length) {
    throw invalid('Incompatible matrix RPC protocol; reload the client and GPU service together');
  }
  if (value.features.includes('program-retain-inputs') && !value.features.includes('program-matmul-f32')) {
    throw invalid('Input retention requires the matrix program feature');
  }
  return Object.freeze({name: value.name, major: value.major, minor: value.minor, legacy: false,
    features: Object.freeze(PROTOCOL.features.filter(feature => value.features.includes(feature)))});
}
