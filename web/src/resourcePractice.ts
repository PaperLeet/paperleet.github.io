import type {Problem} from './types';

/** Editorial connections to introductory concepts, never company interview evidence. */
export const resourcePractice:Record<string,{modules:string[];zh:string;en:string}>={
 'google-ml-crash-course':{modules:['foundations/next_token_loss','architecture/attention'],zh:'结合下一词预测损失和注意力练习，复习课程中的语言模型基础。',en:'Review introductory language-model concepts through next-token loss and attention exercises.'},
 'nvidia-cuda-programming':{modules:['architecture/flash_attention'],zh:'用分块注意力练习理解计算与内存的关系；CUDA 编程请结合官方指南学习。',en:'Explore computation and memory through tiled attention. Learn CUDA programming with the official guide.'},
 'meta-fsdp':{modules:['optimization/adamw'],zh:'先用 AdamW 练习理解优化器状态更新，再结合原文学习分布式分片。',en:'Review optimizer-state updates with AdamW before studying distributed sharding in the original article.'},
 'microsoft-zero-deepspeed':{modules:['optimization/adamw'],zh:'先用 AdamW 练习理解优化器状态更新，再结合原文学习 ZeRO 的分片阶段。',en:'Review optimizer-state updates with AdamW before studying the ZeRO partitioning stages in the original article.'},
};

/** Only uniquely identified, loaded modules can produce native practice links. */
export function resolveResourcePractice(id:string,problems:Problem[]){
 const entry=Object.hasOwn(resourcePractice,id)?resourcePractice[id]:undefined;if(!entry)return;
 const byId=new Map<string,Problem|undefined>(),numbers=new Map<number,number>();
 for(const problem of problems){
  byId.set(problem.id,byId.has(problem.id)?undefined:problem);
  numbers.set(problem.number,(numbers.get(problem.number)||0)+1);
 }
 const modules=entry.modules.flatMap(id=>{const problem=byId.get(id);return problem&&Number.isSafeInteger(problem.number)&&problem.number>0&&numbers.get(problem.number)===1?[problem]:[];});
 return {...entry,modules,missing:entry.modules.length-modules.length,total:entry.modules.length};
}
