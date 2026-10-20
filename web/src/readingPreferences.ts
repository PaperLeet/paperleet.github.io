export type ReadingPreferences={version:1;size:'default'|'18'|'20';spacing:'default'|'relaxed'};
export type ReadingPreferenceSnapshot=Readonly<ReadingPreferences&{saved:boolean}>;
export const READING_PREFERENCES_KEY='paperleet.reading.preferences.v1';
export const defaultReadingPreferences:Readonly<ReadingPreferences>=Object.freeze({version:1,size:'default',spacing:'default'});
const serverSnapshot:ReadingPreferenceSnapshot=Object.freeze({...defaultReadingPreferences,saved:true});
const sizes=['default','18','20'],spacing=['default','relaxed'];

export function parseReadingPreferences(raw:string|null):Readonly<ReadingPreferences>{
 if(!raw||raw.length>512)return defaultReadingPreferences;
 try{
  const value=JSON.parse(raw);
  if(!value||Array.isArray(value)||typeof value!=='object'||Object.keys(value).length!==3||value.version!==1||!sizes.includes(value.size)||!spacing.includes(value.spacing)||!Object.hasOwn(value,'size')||!Object.hasOwn(value,'spacing'))return defaultReadingPreferences;
  return Object.freeze({version:1,size:value.size,spacing:value.spacing});
 }catch{return defaultReadingPreferences;}
}

/** Browser appearance only: no account, draft, grade or content data enters this store. */
export function createReadingPreferencesStore(host?:Window){
 let snapshot:ReadingPreferenceSnapshot=serverSnapshot;
 const listeners=new Set<()=>void>();
 function apply(value:Readonly<ReadingPreferences>){
  const root=host?.document?.documentElement;if(!root)return;
  for(const[key,next]of [['data-reading-size',value.size],['data-reading-spacing',value.spacing]]){
   if(next==='default'){if(root.hasAttribute(key))root.removeAttribute(key);}
   else if(root.getAttribute(key)!==next)root.setAttribute(key,next);
  }
 }
 function publish(value:Readonly<ReadingPreferences>,saved:boolean){
  apply(value);
  if(snapshot.size===value.size&&snapshot.spacing===value.spacing&&snapshot.saved===saved)return;
  snapshot=Object.freeze({...value,saved});for(const listener of listeners)listener();
 }
 function currentValue():ReadingPreferences{return {version:1,size:snapshot.size,spacing:snapshot.spacing};}
 function read(){
  if(!host)return;
  try{publish(parseReadingPreferences(host.localStorage.getItem(READING_PREFERENCES_KEY)),true);}
  catch{publish(currentValue(),false);}
 }
 function persist(value:ReadingPreferences,reset=false){
  let saved=false;
  try{if(host){if(reset)host.localStorage.removeItem(READING_PREFERENCES_KEY);else host.localStorage.setItem(READING_PREFERENCES_KEY,JSON.stringify(value));saved=true;}}catch{/* Apply in this page even when browser storage is unavailable. */}
  publish(value,saved);
 }
 function changed(event:StorageEvent){
  if(event.key!==READING_PREFERENCES_KEY&&event.key!==null)return;
  try{if(event.storageArea!==host?.localStorage)return;}catch{return;}
  // Read the current value instead of replaying a potentially stale event payload.
  read();
 }
 read();host?.addEventListener?.('storage',changed);
 return {
  getSnapshot:()=>snapshot,getServerSnapshot:()=>serverSnapshot,
  subscribe(listener:()=>void){listeners.add(listener);return()=>{listeners.delete(listener);};},
  set(patch:Partial<Pick<ReadingPreferences,'size'|'spacing'>>){
   if(!patch||typeof patch!=='object'||Array.isArray(patch)||Object.keys(patch).some(key=>key!=='size'&&key!=='spacing'))return;
   if(Object.hasOwn(patch,'size')&&!sizes.includes(patch.size!)||Object.hasOwn(patch,'spacing')&&!spacing.includes(patch.spacing!))return;
   const value={...currentValue(),...patch};
   if(snapshot.saved&&value.size===snapshot.size&&value.spacing===snapshot.spacing)return;
   persist(value);
  },
  reset:()=>persist({...defaultReadingPreferences},true),retry:()=>persist(currentValue()),
  destroy(){host?.removeEventListener?.('storage',changed);listeners.clear();},
 };
}

export const readingPreferences=createReadingPreferencesStore(typeof window==='undefined'?undefined:window);
if(import.meta.hot)import.meta.hot.dispose(()=>readingPreferences.destroy());
