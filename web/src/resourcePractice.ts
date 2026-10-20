import type {Problem} from './types';

/** Editorial connections to introductory concepts, never company interview evidence. */
type ResourcePracticeEntry={modules:string[];zh:string;en:string;questions?:{zh:string;en:string}[]};
export const resourcePractice:Record<string,ResourcePracticeEntry>={
 'nvidia-tensorrt-attention':{modules:['architecture/gqa','architecture/flash_attention'],zh:'用 GQA 练习 KV 头共享，再用在线 softmax 理解分块注意力。平台补充练习不包含 TensorRT 内核、分页 KV cache 或请求调度，也不提供吞吐基准。',en:'Connect KV head sharing to GQA and tiled attention to online softmax. These supplementary exercises do not implement TensorRT kernels, paged KV caches or request scheduling, and are not throughput benchmarks.',questions:[{zh:'相同 query 头数下，MHA、GQA 和 MQA 的 KV 头数如何影响缓存大小？',en:'With the same query head count, how do MHA, GQA and MQA KV heads affect cache size?'},{zh:'为什么上下文阶段与逐 token 生成阶段需要不同的计算与内存策略？',en:'Why do context processing and token generation need different computation and memory strategies?'}]},
 'meta-llama3-reference':{modules:['architecture/rms_norm','architecture/rope','architecture/gqa'],zh:'对照历史参考源码中的 RMSNorm、成对 RoPE 旋转与 repeat_kv。练习聚焦独立组件，部分特征统计为平台扩展；不实现完整模型、KV 缓存管理或分布式推理。',en:'Compare RMSNorm, paired RoPE rotations and repeat_kv in the historical reference source. Exercises cover isolated components; partial feature statistics are a PaperLeet extension. They do not implement a full model, KV cache management or distributed inference.',questions:[{zh:'RMSNorm 与 LayerNorm 使用的统计量有什么不同？',en:'How do the statistics used by RMSNorm and LayerNorm differ?'},{zh:'这份实现中，RoPE 旋转、缓存写入与 KV 头重复按什么顺序发生？',en:'In this implementation, in what order do RoPE rotation, cache updates and KV head repetition occur?'}]},
 'google-recommendation-systems':{modules:['retrieval/bpr','retrieval/lightgcn'],zh:'结合 BPR 的成对排序与 LightGCN 的协同传播，练习反馈与嵌入打分；这两种论文方法是平台补充练习。',en:'Practice feedback and embedding scores with BPR pairwise ranking and LightGCN collaborative propagation. These paper methods are supplementary exercises curated by PaperLeet.'},
 'nvidia-gnn-introduction':{modules:['representation/graphsage_mean'],zh:'用 GraphSAGE 均值聚合练习连接邻域采样、消息传递与新节点上的归纳编码。',en:'Connect neighborhood sampling, message passing and inductive encoding on new nodes through the GraphSAGE mean-aggregation exercise.'},
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
