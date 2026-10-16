import {CornerUpLeft} from 'lucide-react';
import {useI18n} from './i18n';
import type {ReadingNotice} from './useReadingPosition';
import './reading-return.css';

export default function ReadingReturnNotice({notice,onRestore,kind='post'}:{notice:ReadingNotice|undefined;onRestore:(opener:HTMLElement)=>void;kind?:'post'|'problem'}){
 const {t}=useI18n();
 if(!notice)return null;
 return <div className="reading-return-notice" role="status"><CornerUpLeft size={14} aria-hidden="true"/><span>{kind==='problem'?(notice==='restored'?t('已回到刚才的收藏题目。','Back at the saved problem you opened.'):notice==='missing'?t('收藏列表已有更新，刚才的题目不在当前范围。','Your saved list changed. The problem is no longer in this view.'):t('收藏列表已加载，可以回到刚才的题目。','Your saved list is ready. Return to the problem you opened.')):notice==='restored'?t('已回到刚才阅读的位置。','Back at your previous reading position.'):notice==='missing'?t('列表已有更新，刚才的帖子暂不在这页。','The list changed. The post is no longer on this page.'):t('列表已加载，可以回到刚才的帖子。','The list is ready. Return to the post you were reading.')}</span>{notice==='available'&&<button type="button" onClick={event=>{event.currentTarget.focus({preventScroll:true});onRestore(event.currentTarget);}}>{t('回到刚才位置','Return to reading position')}</button>}</div>;
}
