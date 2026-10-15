import type { Affiliation, Institution } from './social';

type Organization = { id: string; name: string; name_en: string; domains: string[] };
// Display metadata only: this catalog does not approve registrations or grant verified status.
// Primary sources and asset provenance are recorded in docs/organization-affiliations.md.
const organizations: Organization[] = [
  { id: 'brown', name: '布朗大学', name_en: 'Brown University', domains: ['brown.edu'] },
  { id: 'mit', name: '麻省理工学院', name_en: 'Massachusetts Institute of Technology', domains: ['mit.edu'] },
  { id: 'uc-berkeley', name: '加州大学伯克利分校', name_en: 'University of California, Berkeley', domains: ['berkeley.edu'] },
  { id: 'cornell', name: '康奈尔大学', name_en: 'Cornell University', domains: ['cornell.edu'] },
  { id: 'columbia', name: '哥伦比亚大学', name_en: 'Columbia University', domains: ['columbia.edu'] },
  { id: 'stanford', name: '斯坦福大学', name_en: 'Stanford University', domains: ['stanford.edu'] },
  { id: 'cmu', name: '卡内基梅隆大学', name_en: 'Carnegie Mellon University', domains: ['cmu.edu'] },
  { id: 'harvard', name: '哈佛大学', name_en: 'Harvard University', domains: ['harvard.edu'] },
  { id: 'upenn', name: '宾夕法尼亚大学', name_en: 'University of Pennsylvania', domains: ['upenn.edu'] },
  {"id": "ucla", "name": "加州大学洛杉矶分校", "name_en": "University of California, Los Angeles", "domains": ["ucla.edu"]},
  {"id": "ucsd", "name": "加州大学圣迭戈分校", "name_en": "University of California San Diego", "domains": ["ucsd.edu"]},
  {"id": "uiuc", "name": "伊利诺伊大学厄巴纳–香槟分校", "name_en": "University of Illinois Urbana-Champaign", "domains": ["illinois.edu"]},
  {"id": "umass-amherst", "name": "马萨诸塞大学阿默斯特分校", "name_en": "University of Massachusetts Amherst", "domains": ["umass.edu"]},
  {"id": "uwashington", "name": "华盛顿大学", "name_en": "University of Washington", "domains": ["washington.edu", "uw.edu"]},
  {"id": "usc", "name": "南加州大学", "name_en": "University of Southern California", "domains": ["usc.edu"]},
  {"id": "umd", "name": "马里兰大学", "name_en": "University of Maryland", "domains": ["umd.edu"]},
  {"id": "ut-austin", "name": "得克萨斯大学奥斯汀分校", "name_en": "The University of Texas at Austin", "domains": ["utexas.edu"]},
  {"id": "nyu", "name": "纽约大学", "name_en": "New York University", "domains": ["nyu.edu"]},
  {"id": "princeton", "name": "普林斯顿大学", "name_en": "Princeton University", "domains": ["princeton.edu"]},
  {"id": "tsinghua", "name": "清华大学", "name_en": "Tsinghua University", "domains": ["tsinghua.edu.cn"]},
  {"id": "pku", "name": "北京大学", "name_en": "Peking University", "domains": ["pku.edu.cn"]},
  {"id": "hku", "name": "香港大学", "name_en": "The University of Hong Kong", "domains": ["hku.hk"]},
  {"id": "cuhk", "name": "香港中文大学", "name_en": "The Chinese University of Hong Kong", "domains": ["cuhk.edu.hk"]},
  {"id": "hkust", "name": "香港科技大学", "name_en": "The Hong Kong University of Science and Technology", "domains": ["hkust.edu.hk", "ust.hk"]},
  {"id": "nus", "name": "新加坡国立大学", "name_en": "National University of Singapore", "domains": ["nus.edu.sg"]},
  {"id": "ntu", "name": "南洋理工大学", "name_en": "Nanyang Technological University", "domains": ["ntu.edu.sg"]},
  { id: 'openai', name: 'OpenAI', name_en: 'OpenAI', domains: ['openai.com'] },
  { id: 'google', name: 'Google', name_en: 'Google', domains: ['google.com'] },
  { id: 'google', name: 'Google DeepMind', name_en: 'Google DeepMind', domains: ['deepmind.com', 'deepmind.google'] },
  { id: 'meta', name: 'Meta', name_en: 'Meta', domains: ['meta.com', 'facebook.com'] },
  { id: 'amazon', name: 'Amazon', name_en: 'Amazon', domains: ['amazon.com'] },
  { id: 'microsoft', name: '微软', name_en: 'Microsoft', domains: ['microsoft.com'] },
  { id: 'apple', name: 'Apple', name_en: 'Apple', domains: ['apple.com'] },
  { id: 'nvidia', name: 'NVIDIA', name_en: 'NVIDIA', domains: ['nvidia.com'] },
  { id: 'anthropic', name: 'Anthropic', name_en: 'Anthropic', domains: ['anthropic.com'] },
  { id: 'bytedance', name: '字节跳动', name_en: 'ByteDance', domains: ['bytedance.com'] },
  { id: 'alibaba', name: '阿里巴巴', name_en: 'Alibaba', domains: ['alibaba.com', 'alibabagroup.com'] },
  { id: 'tencent', name: '腾讯', name_en: 'Tencent', domains: ['tencent.com'] },
  { id: 'baidu', name: '百度', name_en: 'Baidu', domains: ['baidu.com'] },
  { id: 'meituan', name: '美团', name_en: 'Meituan', domains: ['meituan.com'] },
  { id: 'kuaishou', name: '快手', name_en: 'Kuaishou', domains: ['kuaishou.com'] },
  { id: 'deepseek', name: 'DeepSeek', name_en: 'DeepSeek', domains: ['deepseek.com'] },
  { id: 'xai', name: 'xAI', name_en: 'xAI', domains: ['x.ai'] },
  { id: 'tesla', name: 'Tesla', name_en: 'Tesla', domains: ['tesla.com'] },
];

function normalizeDomain(value: string | undefined): string | undefined {
  const domain = value?.trim().toLowerCase().replace(/\.$/, '');
  return domain && domain.length <= 253 && /^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,63}$/.test(domain) ? domain : undefined;
}

function friendlyDomain(domain: string): string {
  const labels = domain.split('.');
  const category = labels.at(-2)!;
  const compoundSuffix = labels.at(-1)!.length === 2 && ['ac', 'edu', 'com', 'co', 'org', 'net', 'gov'].includes(category);
  const label = labels.at(compoundSuffix ? -3 : -2) || labels[0];
  return label.split('-').map(word => word.length <= 3 ? word.toUpperCase() : word[0].toUpperCase() + word.slice(1)).join(' ');
}

function isName(value: string | undefined): value is string {
  return !!value?.trim() && !/@|:\/\/|^\/\//.test(value) && !normalizeDomain(value.split(/[/?#]/, 1)[0]);
}

export function resolveAffiliation(affiliation?: Affiliation | null, institution?: Institution | null): Institution | undefined {
  const domain = normalizeDomain(affiliation?.domain);
  const known = domain ? organizations.find(org => org.domains.some(base => domain === base || domain.endsWith(`.${base}`))) : organizations.find(org => org.id === institution?.id);
  if (affiliation && isName(affiliation.name)) return {
    id: known?.id || institution?.id || '', name: affiliation.name.trim(),
    name_en: isName(affiliation.name_en) ? affiliation.name_en.trim() : known?.name_en,
  };
  if (known) return { id: known.id, name: known.name, name_en: known.name_en };
  if (institution && isName(institution.name)) return {
    id: institution.id, name: institution.name.trim(), name_en: isName(institution.name_en) ? institution.name_en.trim() : undefined,
  };
  if (domain) return { id: '', name: friendlyDomain(domain) };
  return undefined;
}
