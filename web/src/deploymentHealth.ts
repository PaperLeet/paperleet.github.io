export type DeploymentHealth = {
  mode:'local'|'hosted'; judge:{ready:boolean;reason?:string}; auth_required:boolean;
};
const browserHealth = ():DeploymentHealth => ({mode:'local',auth_required:false,judge:{ready:false}});

/** A standalone browser workspace does not depend on an optional native judge. */
export function initialDeploymentHealth(hostedEdition:boolean) {
  return hostedEdition ? undefined : browserHealth();
}
export function unavailableDeploymentHealth(previous:DeploymentHealth|undefined, hostedEdition:boolean) {
  if (!hostedEdition) return browserHealth();
  // Never turn an unresolved hosted account into a local workspace on failure.
  return previous ? {...previous,judge:{ready:false,reason:'Service unavailable'}} : undefined;
}
