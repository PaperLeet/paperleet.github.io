type SearchableModule={id:string;number?:number|null;title:string;title_en?:string|null;translations?:{en?:{title?:string}}};
export const relatedModuleQuery=(value:string)=>Array.from(value.replace(/[\u0000-\u001f\u007f]/g,' ').trim()).slice(0,100).join('');
const fold=(value:string)=>value.normalize('NFKC').toLowerCase();
/** Search only loaded public labels, numbers and canonical IDs; never alter a selection. */
export function searchRelatedModules<T extends SearchableModule>(modules:T[],query:string):T[]{
 const terms=fold(relatedModuleQuery(query)).split(/\s+/u).filter(Boolean);
 if(!terms.length)return modules;
 return modules.filter(module=>{
  const text=fold([module.number?`#${module.number} · ${module.number}.`:'',module.id,module.title,module.title_en,module.translations?.en?.title].join('\n'));
  return terms.every(term=>text.includes(term));
 });
}
