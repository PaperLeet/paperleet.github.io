import type {ProblemTopic} from './communityTopics';
import {paperIdentity} from './paperRegistrySearch';

export type PracticeMention={start:number;end:number;topic:ProblemTopic};
export type MentionIndex={pattern:RegExp;topics:Map<string,ProblemTopic>;sources:Map<string,ProblemTopic>};
const aliases:Record<string,string[]>={
 'architecture/attention':['scaled dot-product attention','缩放点积注意力','Transformer','Attention Is All You Need'],
 'architecture/multihead_attention':['multi-head attention','multihead attention','MHA','多头注意力'],
 'architecture/flash_attention':['FlashAttention','Flash Attention','Flash-Attention'],
 'architecture/layer_norm':['LayerNorm','Layer Normalization','层归一化'],
 'architecture/rms_norm':['RMSNorm','RMS Normalization'],
 'architecture/rope':['RoPE','rotary position embedding','旋转位置编码'],
 'architecture/gqa':['GQA','grouped-query attention','分组查询注意力'],
 'optimization/adamw':['AdamW'],
 'finetuning/lora':['LoRA','low-rank adaptation'],
 'foundations/top_p_sampling':['Top-p','Top p','nucleus sampling','核采样'],
 'foundations/top_k_sampling':['Top-k','Top k'],
 'foundations/next_token_loss':['next-token prediction','next token prediction','下一词预测'],
 'foundations/sft_loss_mask':['SFT','监督微调'],
 'foundations/multiturn_sft_mask':['multi-turn SFT','多轮 SFT'],
 'foundations/gradient_accumulation':['gradient accumulation','梯度累积'],
 'foundations/gradient_clipping':['gradient clipping','梯度裁剪'],
 'architecture/batch_norm':['BatchNorm','batch normalization','批归一化'],
 'architecture/group_norm':['GroupNorm','group normalization','组归一化'],
 'architecture/residual_block':['ResNet','residual network','残差网络'],
 'representation/vae':['VAE','variational autoencoder','变分自编码器'],
 'representation/graphsage_mean':['GraphSAGE','Graph SAGE'],
 'rl/ppo_clip':['PPO','proximal policy optimization'],
 'rl/gae':['GAE','generalized advantage estimation','广义优势估计'],
 'rl/dpo':['DPO','direct preference optimization','直接偏好优化'],
 'rl/dr_grpo':['Dr. GRPO','Dr GRPO','Dr.GRPO'],
 'rl/grpo':['GRPO'],
 'finetuning/ewc':['EWC','elastic weight consolidation'],
};
const key=(value:string)=>value.toLowerCase().replace(/[‐‑–—]/g,'-').replace(/\s+/g,' ').trim();
const escape=(value:string)=>value.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
const word=(value:string|undefined)=>!!value&&/[a-z0-9_]/i.test(value);

/** Exact maintained names and unique exercise numbers only. Generic tags and fuzzy matches never create an exercise claim. */
export function practiceMentionIndex(source:ProblemTopic[]):MentionIndex|undefined{
 const active=source.filter(topic=>!topic.retired),byId=new Map(active.map(topic=>[topic.id,topic]));
 const candidates=new Map<string,Map<string,ProblemTopic>>(),numbers=new Map<number,Map<string,ProblemTopic>>();
 const add=(name:string|undefined,topic:ProblemTopic)=>{if(!name||name.trim().length<3||name.length>220)return;const term=key(name),items=candidates.get(term)||new Map();items.set(topic.id,topic);candidates.set(term,items);};
 for(const topic of active){
   if(typeof topic.number==='number'&&Number.isSafeInteger(topic.number)&&topic.number>0&&topic.number<=99_999_999){const items=numbers.get(topic.number)||new Map();items.set(topic.id,topic);numbers.set(topic.number,items);}
   for(const name of [topic.title,topic.title_en]){
     add(name||undefined,topic);
     if(name){const prefix=name.split(/[:：]/)[0];if(prefix!==name)add(prefix,topic);const acronym=/^([A-Za-z][A-Za-z0-9.-]*)\s/.exec(name)?.[1];if(acronym&&(/[A-Z].*[A-Z]/.test(acronym)||/\d/.test(acronym)))add(acronym,topic);}
   }
 }
 const topics=new Map<string,ProblemTopic>(),sources=new Map<string,ProblemTopic>();
 for(const [term,items] of candidates)if(items.size===1)topics.set(term,[...items.values()][0]);
 // A paper with several exercises opens its first maintained module.
 for(const topic of [...active].sort((a,b)=>(a.number||Infinity)-(b.number||Infinity))){
   if(topic.paper?.title&&!topics.has(key(topic.paper.title)))topics.set(key(topic.paper.title),topic);
   for(const name of topic.paper_short_title?.split(/\s+[·/]\s+/)||[])if(name.trim().length>=3&&!topics.has(key(name)))topics.set(key(name),topic);
   for(const value of [topic.paper?.arxiv,topic.paper?.url])if(value&&!sources.has(paperIdentity(value)))sources.set(paperIdentity(value),topic);
 }
 for(const [id,names] of Object.entries(aliases)){const topic=byId.get(id);if(topic)for(const name of names)topics.set(key(name),topic);}
 for(const [number,items]of numbers){if(items.size===1)topics.set('#'+number,[...items.values()][0]);else topics.delete('#'+number);}
 if(!topics.size)return;
 const pattern=new RegExp([...topics.keys()].sort((a,b)=>b.length-a.length).map(term=>escape(term).replace(/-/g,'[-‐‑–—]').replace(/ /g,'\\s+')).join('|'),'gi');
 return {pattern,topics,sources};
}

export function findPracticeMentions(text:string,index?:MentionIndex):PracticeMention[]{
 if(!index)return [];
 const matches:PracticeMention[]=[];index.pattern.lastIndex=0;
 for(const match of text.matchAll(index.pattern)){
   const start=match.index!,end=start+match[0].length,term=key(match[0]);
   if(match[0][0]==='#'&&(word(text[start-1])||/[#@/]/.test(text[start-1]||'')||/^[.\/-][a-z0-9_]/i.test(text.slice(end))))continue;
   if(word(match[0][0])&&word(text[start-1])||word(match[0].at(-1))&&word(text[end]))continue;
   // These acronyms are also ordinary English/programming words.
   if(['clip','var','jit'].includes(term)&&match[0]!==match[0].toUpperCase())continue;
   const topic=index.topics.get(term);if(topic)matches.push({start,end,topic});
 }
 return matches;
}

type Node={type:string;value?:string;children?:Node[];url?:string;data?:unknown};
const excluded=new Set(['link','linkReference','image','imageReference','code','inlineCode','math','inlineMath','html','definition']);

/** Walk Markdown text, leaving code, math and author-supplied links untouched. */
export function transformPracticeMentions(tree:Node,index:MentionIndex|undefined,href:(topic:ProblemTopic)=>string,onMatch?:(topic:ProblemTopic)=>void){
 // A citation remains the author's source link; its exercise joins the summary.
 if(tree.type==='link'&&tree.url){const topic=index?.sources.get(paperIdentity(tree.url));if(topic)onMatch?.(topic);}
 if(excluded.has(tree.type)||!tree.children)return;
 tree.children=tree.children.flatMap(child=>{
   if(child.type!=='text'||!child.value){transformPracticeMentions(child,index,href,onMatch);return [child];}
   const text=child.value,matches=findPracticeMentions(text,index);if(!matches.length)return [child];
   const output:Node[]=[];let cursor=0;
   for(const match of matches){
     if(match.start>cursor)output.push({type:'text',value:text.slice(cursor,match.start)});
     onMatch?.(match.topic);
     output.push({type:'link',url:href(match.topic),data:{hProperties:{className:'practice-mention-link','data-practice-mention':match.topic.id}},children:[{type:'text',value:text.slice(match.start,match.end)}]});cursor=match.end;
   }
   if(cursor<text.length)output.push({type:'text',value:text.slice(cursor)});
   return output;
 });
}
