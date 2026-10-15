import { useState } from 'react';
import { Building2 } from 'lucide-react';
import './organization-logo.css';

// Reviewed, bundled assets only. Never turn a user's email domain into a remote image request.
const files: Record<string, string> = {
  openai: 'openai-linkedin.jpg', microsoft: 'microsoft-linkedin.jpg', tesla: 'tesla-linkedin.jpg',
  bytedance: 'bytedance-linkedin.jpg', alibaba: 'alibaba-linkedin.jpg', tencent: 'tencent-linkedin.jpg', baidu: 'baidu-linkedin.jpg',
  meituan: 'meituan-linkedin.jpg', kuaishou: 'kuaishou-linkedin.jpg', deepseek: 'deepseek-linkedin.jpg', brown: 'brown.png',
  google: 'google-linkedin.jpg', meta: 'meta-linkedin.jpg', amazon: 'amazon-linkedin.jpg', apple: 'apple-linkedin.jpg',
  nvidia: 'nvidia-linkedin.jpg', anthropic: 'anthropic-linkedin.jpg', xai: 'xai.ico',
  mit: 'mit.svg', 'uc-berkeley': 'uc-berkeley.ico', cornell: 'cornell.ico',
  stanford: 'stanford.ico', cmu: 'cmu.ico',
};

export default function OrganizationLogo({ id, size = 16, className = '' }: { id: string; size?: number; className?: string }) {
  const file = Object.hasOwn(files, id) ? files[id] : undefined;
  const [failed, setFailed] = useState<string>();
  return <span className={`organization-logo ${className}`} style={{ width: size, height: size }} aria-hidden="true">
    {file && failed !== file
      ? <img src={`${import.meta.env.BASE_URL}organization-logos/${file}`} alt="" width={size} height={size} loading="lazy" decoding="async" referrerPolicy="no-referrer" onError={() => setFailed(file)} />
      : <Building2 size={size} strokeWidth={1.7} />}
  </span>;
}
