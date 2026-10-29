import { useEffect, useId, useRef, useState } from 'react';
import { ArrowUpRight, Download, MessageCircle, QrCode, X } from 'lucide-react';
import { useI18n } from './i18n';
import './group-chat.css';

const discordInvite = 'https://discord.gg/36PZ3WRxn';
const wechatImage = `${import.meta.env.BASE_URL}community/wechat-group-2026-11-02.png`;
// The supplied invitation says it is valid through November 2, 2026.
const wechatLastDay = [2026, 10, 2] as const;

export default function GroupChat({ route }: { route: string }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  const title = useId();
  const expired = Date.now() >= new Date(wechatLastDay[0], wechatLastDay[1], wechatLastDay[2] + 1).getTime();
  useEffect(() => { setOpen(false); }, [route]);
  useEffect(() => {
    const node = dialog.current;
    if (open && node && !node.open) node.showModal();
    else if (!open && node?.open) node.close();
  }, [open]);
  const label = t('加入群聊', 'Join group chat');
  return <>
    <button type="button" className="group-chat-trigger" title={label} aria-label={label}
      aria-haspopup="dialog" aria-expanded={open} onClick={() => { setImageFailed(false); setOpen(true); }}>
      <MessageCircle size={18} aria-hidden="true" /><span>{t('加入群聊', 'Group chat')}</span>
    </button>
    <dialog ref={dialog} className="group-chat-dialog" aria-labelledby={title}
      onCancel={() => setOpen(false)} onClose={() => setOpen(false)}
      onClick={event => {
        if (event.target !== event.currentTarget) return;
        const box = event.currentTarget.getBoundingClientRect();
        if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) setOpen(false);
      }}>
      {open && <>
        <header>
          <h2 id={title}>{label}</h2>
          <button type="button" className="group-chat-close" aria-label={t('关闭群聊窗口', 'Close group chat')}
            onClick={() => setOpen(false)} autoFocus><X size={20} aria-hidden="true" /></button>
        </header>
        <p className="group-chat-intro">{t('一起交流论文、算法与研究。', 'Discuss papers, algorithms, and research.')}</p>
        <a className="group-chat-discord" href={discordInvite} target="_blank" rel="noopener noreferrer">
          <span className="group-chat-discord-icon" aria-hidden="true"><MessageCircle size={22} /></span>
          <span><strong>Discord</strong><small>{t('加入 PaperLeet 社群', 'Join the PaperLeet community')}</small></span>
          <ArrowUpRight size={20} aria-hidden="true" />
        </a>
        <section className="group-chat-wechat" aria-label={t('微信群', 'WeChat group')}>
          <h3><QrCode size={19} aria-hidden="true" />{t('微信群', 'WeChat group')}</h3>
          {expired ? <p className="group-chat-unavailable" role="status">{t('本期二维码已过期，更新前请先加入 Discord。', 'This QR code has expired. Join Discord while we refresh the WeChat invitation.')}</p> : <>
            <p>{t('微信扫码加入，或保存二维码后识别。', 'Scan or save this code to join on WeChat.')}</p>
            {imageFailed ? <p className="group-chat-unavailable" role="status">{t('二维码加载失败，请打开原图或稍后重试。', 'The QR code could not load. Open the original image or try again later.')}</p>
              : <div className="group-chat-qr"><img src={wechatImage} width="720" height="1140"
                alt={t('PaperLeet 微信群二维码，有效期至 2026 年 11 月 2 日', 'PaperLeet WeChat group QR code, valid through November 2, 2026')}
                onError={() => setImageFailed(true)} /></div>}
            <p className="group-chat-expiry">{t('有效期至 2026 年 11 月 2 日', 'Valid through November 2, 2026')}</p>
            <div className="group-chat-image-actions">
              <a href={wechatImage} download="paperleet-wechat-group.png"><Download size={15} aria-hidden="true" />{t('保存二维码', 'Save QR code')}</a>
              <a href={wechatImage} target="_blank" rel="noopener noreferrer">{t('打开原图', 'Open original')}<ArrowUpRight size={15} aria-hidden="true" /></a>
            </div>
          </>}
        </section>
      </>}
    </dialog>
  </>;
}
