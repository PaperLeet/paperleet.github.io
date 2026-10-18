import AppLink from './AppLink';
import { ArrowLeft, ArrowRight, Award, Check, Globe2, HardDriveDownload, Moon, Palette, Sun, UserRound } from 'lucide-react';
import { useI18n } from './i18n';
import './settings.css';

export type SettingsProps = {
  theme: 'light' | 'dark';
  onTheme: (theme: 'light' | 'dark') => void;
  returnTo: string;
  onBackup: () => void;
  backupReady: boolean;
  workspaceName: string;
  accountHref?: string;
};

export default function SettingsPage({ theme, onTheme, returnTo, onBackup, backupReady, workspaceName, accountHref }: SettingsProps) {
  const { locale, setLocale, t } = useI18n();
  return <main className="settings-page">
    <AppLink className="back-link settings-back" href={returnTo}><ArrowLeft size={15} aria-hidden="true" />{t('返回刚才的页面', 'Back to where you left off')}</AppLink>
    <header className="settings-heading"><h1>{t('设置', 'Settings')}</h1></header>
    <section className="settings-card" aria-labelledby="settings-preferences-title">
      <div className="settings-card-heading"><span className="settings-section-icon"><Palette size={19} aria-hidden="true" /></span><div><h2 id="settings-preferences-title">{t('使用偏好', 'Preferences')}</h2><p>{t('这些偏好会自动保存在当前浏览器。', 'Preferences save automatically in this browser.')}</p></div></div>
      <div className="settings-row">
        <div><h3>{t('外观', 'Appearance')}</h3></div>
        <div className="settings-choice" role="group" aria-label={t('外观', 'Appearance')}>
          <button type="button" aria-pressed={theme === 'light'} onClick={() => onTheme('light')}><Sun size={16} aria-hidden="true" />{t('浅色', 'Light')}{theme === 'light' && <Check size={13} aria-hidden="true" />}</button>
          <button type="button" aria-pressed={theme === 'dark'} onClick={() => onTheme('dark')}><Moon size={16} aria-hidden="true" />{t('深色', 'Dark')}{theme === 'dark' && <Check size={13} aria-hidden="true" />}</button>
        </div>
      </div>
      <div className="settings-row">
        <div><h3>{t('界面语言', 'Language')}</h3><p>{t('题面提供翻译时会同步切换。', 'Translated statements follow your language choice.')}</p></div>
        <div className="settings-choice" role="group" aria-label={t('界面语言', 'Language')}>
          <button type="button" lang="zh-CN" aria-pressed={locale === 'zh'} onClick={() => setLocale('zh')}><Globe2 size={15} aria-hidden="true" />中文{locale === 'zh' && <Check size={13} aria-hidden="true" />}</button>
          <button type="button" lang="en" aria-pressed={locale === 'en'} onClick={() => setLocale('en')}>English{locale === 'en' && <Check size={13} aria-hidden="true" />}</button>
        </div>
      </div>
    </section>
    <section className="settings-card settings-backup-card" aria-labelledby="settings-backup-title">
      <div className="settings-card-heading"><span className="settings-section-icon"><HardDriveDownload size={19} aria-hidden="true" /></span><div><h2 id="settings-backup-title">{t('代码备份与恢复', 'Code backup & restore')}</h2><p>{workspaceName}</p></div></div>
      <p className="settings-backup-description">{t('导出所有练习代码，或从备份中核对并恢复。换设备、清理浏览器数据前，可以先在这里保留一份。', 'Export your practice code, or review and restore a backup. Keep a copy before switching devices or clearing browser data.')}</p>
      <div className="settings-card-actions"><span>{t('仅包含代码，不导入成绩或账号资料。', 'Code only. Grades and account details are not imported.')}</span><button className="button subtle" type="button" aria-haspopup="dialog" disabled={!backupReady} onClick={onBackup}>{t('管理代码备份', 'Manage code backups')}<ArrowRight size={15} aria-hidden="true" /></button></div>
    </section>
    {accountHref && <AppLink className="settings-resource" href={accountHref}><span className="settings-section-icon"><UserRound size={19} aria-hidden="true" /></span><span><b>{t('个人资料与隐私', 'Profile & privacy')}</b></span><ArrowRight size={16} aria-hidden="true" /></AppLink>}
    <AppLink className="settings-resource" href="#/achievements"><span className="settings-section-icon"><Award size={19} aria-hidden="true" /></span><span><b>{t('学习勋章', 'Learning achievements')}</b></span><ArrowRight size={16} aria-hidden="true" /></AppLink>
  </main>;
}
