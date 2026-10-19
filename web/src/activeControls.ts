/** Native controls in retained but hidden/inert surfaces cannot start work. */
export function controlIsActive(node:HTMLElement|null|undefined){
 return !!node?.isConnected&&!node.closest('[hidden],[inert],[aria-hidden="true"]');
}
