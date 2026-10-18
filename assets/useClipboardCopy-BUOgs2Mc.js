import{c as h}from"./index-CdVANPBM.js";import{r as e}from"./index-CO9dNzeH.js";/**
 * @license lucide-react v0.468.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const E=h("ChevronUp",[["path",{d:"m18 15-6-6-6 6",key:"153udz"}]]);/**
 * @license lucide-react v0.468.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const k=h("WrapText",[["line",{x1:"3",x2:"21",y1:"6",y2:"6",key:"4m8b97"}],["path",{d:"M3 12h15a3 3 0 1 1 0 6h-4",key:"1cl7v7"}],["polyline",{points:"16 16 14 18 16 20",key:"1jznyi"}],["line",{x1:"3",x2:"10",y1:"18",y2:"18",key:"1h33wv"}]]);function C(){const[u,n]=e.useState("idle"),[v,o]=e.useState(),s=e.useRef(!1),a=e.useRef(!1),f=e.useRef(0),l=e.useRef(void 0),t=e.useRef("");function d(){f.current++,a.current=!1,window.clearTimeout(l.current),l.current=void 0}e.useEffect(()=>(s.current=!0,t.current=location.href,()=>{s.current=!1,d()}),[]),e.useEffect(()=>{if(u==="idle")return;const r=()=>{t.current!==location.href&&(t.current=location.href,d(),n("idle"),o(void 0))};return window.addEventListener("hashchange",r),window.addEventListener("popstate",r),()=>{window.removeEventListener("hashchange",r),window.removeEventListener("popstate",r)}},[u]);function y(r){if(a.current)return;a.current=!0,n("copying"),o(void 0);const w=++f.current,p=location.href;t.current=p;const i=c=>{!s.current||f.current!==w||(d(),t.current=location.href,location.href===p?n(c):(n("idle"),o(void 0)))};try{const c=r();o(c),l.current=window.setTimeout(()=>i("failed"),8e3),Promise.resolve(navigator.clipboard.writeText(c)).then(()=>i("copied"),()=>i("failed"))}catch{i("failed")}}return{status:u,value:v,copy:y}}export{E as C,k as W,C as u};
