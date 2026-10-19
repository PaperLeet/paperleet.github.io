import {Prec,EditorState,Facet} from '@codemirror/state';
import {EditorView,keymap,runScopeHandlers,type Panel,type ViewUpdate} from '@codemirror/view';
import {SearchQuery,search,searchKeymap,getSearchQuery,setSearchQuery,openSearchPanel,closeSearchPanel,findNext,findPrevious,replaceNext,replaceAll,selectMatches,gotoLine} from '@codemirror/search';
import {indentWithTab} from '@codemirror/commands';

const panels=new WeakMap<EditorView,EditorSearchPanel>();
type Option='caseSensitive'|'wholeWord'|'regexp';
const phrases:Record<string,string>={Find:'查找',Replace:'替换','Match case':'区分大小写','Match whole word':'全词匹配','Use regular expression':'使用正则表达式','Previous match':'上一个匹配','Next match':'下一个匹配','Close find':'关闭查找','Toggle replace':'展开或收起替换','Replace next':'替换当前匹配','Replace all':'全部替换','No results':'无匹配结果','Invalid regular expression':'正则表达式无效',matches:'个匹配','Find in code':'在代码中查找','Replace in code':'替换代码中的匹配','go to line':'跳转到行','Go to line':'跳转到行',go:'跳转'};
const searchLocale=Facet.define<string,string>({combine:values=>values[0]||'en'});
export const editorSearchLocale=(locale:string)=>searchLocale.of(locale);

/** Keep query, history, matching and replacement in CodeMirror's own transactions. */
class EditorSearchPanel implements Panel {
  dom:HTMLDivElement;
  top=true;
  private input:HTMLInputElement;
  private replacement:HTMLInputElement;
  private replacementRow:HTMLDivElement;
  private count:HTMLSpanElement;
  private error:HTMLDivElement;
  private toggle:HTMLButtonElement;
  private options:Record<Option,HTMLButtonElement>;
  private replaceButtons:HTMLButtonElement[]=[];
  private labels=new Map<HTMLElement,string>();
  private cachedDoc?:EditorState['doc'];
  private cachedQuery?:SearchQuery;
  private matches:{from:number;to:number}[]=[];
  private expanded=false;
  private anchor:number;
  constructor(private view:EditorView) {
    const doc=view.dom.ownerDocument;this.anchor=view.state.selection.main.from;
    const row=()=>{const node=doc.createElement('div');node.className='editor-find-row';return node;};
    const button=(text:string,label:string,action:()=>void)=>{
      const node=doc.createElement('button');node.type='button';node.textContent=text;this.labels.set(node,label);
      node.addEventListener('click',action);return node;
    };
    const field=(label:string)=>{const node=doc.createElement('input');node.type='text';node.spellcheck=false;node.autocomplete='off';this.labels.set(node,label);node.addEventListener('input',()=>this.commit());return node;};
    this.dom=doc.createElement('div');this.dom.className='cm-paperleet-search';this.dom.setAttribute('role','search');
    this.toggle=button('›','Toggle replace',()=>this.setExpanded(!this.expanded));this.toggle.className='editor-find-expand';
    this.input=field('Find in code');this.input.setAttribute('main-field','true');
    this.count=doc.createElement('span');this.count.className='editor-find-count';this.count.setAttribute('role','status');this.count.setAttribute('aria-live','polite');
    this.options={caseSensitive:button('Aa','Match case',()=>this.flip('caseSensitive')),wholeWord:button('ab','Match whole word',()=>this.flip('wholeWord')),regexp:button('.*','Use regular expression',()=>this.flip('regexp'))};
    this.options.wholeWord.className='editor-find-whole-word';
    const find=row();find.append(this.toggle,this.input,...Object.values(this.options),this.count,button('↑','Previous match',()=>findPrevious(view)),button('↓','Next match',()=>findNext(view)),button('×','Close find',()=>closeSearchPanel(view)));
    this.replacement=field('Replace in code');
    this.replacementRow=row();this.replacementRow.classList.add('editor-replace-row');this.replacementRow.hidden=true;
    this.replaceButtons=[button('↵','Replace next',()=>replaceNext(view)),button('⇉','Replace all',()=>replaceAll(view))];
    this.replacementRow.append(this.replacement,...this.replaceButtons);
    this.error=doc.createElement('div');this.error.className='editor-find-error';this.error.setAttribute('role','status');this.error.hidden=true;
    this.dom.append(find,this.replacementRow,this.error);
    this.dom.addEventListener('keydown',event=>this.keydown(event));
    panels.set(view,this);this.sync();
  }
  mount(){this.input.focus();this.input.select();}
  destroy(){panels.delete(this.view);}
  get replacementOpen(){return this.expanded;}
  setExpanded(expanded:boolean){this.expanded=expanded;this.replacementRow.hidden=!expanded;this.toggle.setAttribute('aria-expanded',String(expanded));this.toggle.textContent=expanded?'⌄':'›';}
  private flip(option:Option){const query=getSearchQuery(this.view.state);this.view.dispatch({effects:setSearchQuery.of(new SearchQuery({...query,[option]:!query[option]}))});}
  private commit(){
    const previous=getSearchQuery(this.view.state),query=new SearchQuery({...previous,search:this.input.value,replace:this.replacement.value});
    this.view.dispatch({effects:setSearchQuery.of(query)});
    if(query.valid&&query.search!==previous.search){
      const state=this.view.state,start=Math.min(this.anchor,state.doc.length);let match=query.getCursor(state,start).next();
      if(match.done)match=query.getCursor(state,0,start).next();
      if(!match.done)this.view.dispatch({selection:{anchor:match.value.from,head:match.value.to},effects:EditorView.scrollIntoView(match.value.from,{y:'center'})});
    }
  }
  private keydown(event:KeyboardEvent){
    if(event.isComposing||event.keyCode===229)return;
    if(runScopeHandlers(this.view,event,'search-panel')){event.preventDefault();event.stopPropagation();return;}
    if(event.key==='Enter'&&!event.altKey&&!event.metaKey&&!event.ctrlKey){event.preventDefault();event.stopPropagation();if(event.target===this.replacement)replaceNext(this.view);else(event.shiftKey?findPrevious:findNext)(this.view);}
  }
  update(update:ViewUpdate){if(update.docChanged||update.selectionSet||update.transactions.length)this.sync();}
  private sync(){
    const state=this.view.state,query=getSearchQuery(state),phrase=(key:string)=>state.facet(searchLocale)==='zh'?phrases[key]||key:key;
    this.dom.setAttribute('aria-label',phrase('Find'));
    for(const [node,label]of this.labels){node.setAttribute('aria-label',phrase(label));node.title=phrase(label);}
    this.input.placeholder=phrase('Find');this.replacement.placeholder=phrase('Replace');
    if(this.input.value!==query.search)this.input.value=query.search;
    if(this.replacement.value!==query.replace)this.replacement.value=query.replace;
    for(const key of Object.keys(this.options)as Option[])this.options[key].setAttribute('aria-pressed',String(query[key]));
    this.toggle.setAttribute('aria-expanded',String(this.expanded));
    const invalid=!!query.search&&!query.valid;
    this.input.setAttribute('aria-invalid',String(invalid));this.error.hidden=!invalid;this.error.textContent=invalid?phrase('Invalid regular expression'):'';
    if(state.doc!==this.cachedDoc||!this.cachedQuery?.eq(query)){
      this.cachedDoc=state.doc;this.cachedQuery=query;this.matches=[];
      if(query.valid){const cursor=query.getCursor(state);while(this.matches.length<=1000){const match=cursor.next();if(match.done)break;this.matches.push({from:match.value.from,to:match.value.to});}}
    }
    const selection=state.selection.main,index=this.matches.findIndex(match=>match.from===selection.from&&match.to===selection.to);
    this.count.textContent=!query.search||invalid?'':this.matches.length>1000?'1000+ '+phrase('matches'):!this.matches.length?phrase('No results'):`${index<0?'—':index+1} / ${this.matches.length}`;
    this.count.dataset.empty=String(!!query.search&&!invalid&&!this.matches.length);
    this.replacement.disabled=state.readOnly;
    for(const button of this.replaceButtons)button.disabled=state.readOnly||!query.valid||!this.matches.length;
  }
}
export function openEditorFind(view:EditorView,replace=false){openSearchPanel(view);panels.get(view)?.setExpanded(replace);return true;}
const toggleOption=(key:Option)=>(view:EditorView)=>{const query=getSearchQuery(view.state);view.dispatch({effects:setSearchQuery.of(new SearchQuery({...query,[key]:!query[key]}))});return true;};
export const editorSearch=[search({top:true,literal:true,createPanel:view=>new EditorSearchPanel(view)}),Prec.high(keymap.of([
  {key:'Mod-f',run:view=>openEditorFind(view),scope:'editor search-panel'},
  {key:'Ctrl-f',run:view=>openEditorFind(view),scope:'editor search-panel'},
  {key:'Mod-h',mac:'Mod-Alt-f',run:view=>openEditorFind(view,true),scope:'editor search-panel'},
  {key:'Ctrl-g',run:gotoLine},
  {key:'Alt-c',mac:'Mod-Alt-c',run:toggleOption('caseSensitive'),scope:'search-panel'},
  {key:'Alt-w',mac:'Mod-Alt-w',run:toggleOption('wholeWord'),scope:'search-panel'},
  {key:'Alt-r',mac:'Mod-Alt-r',run:toggleOption('regexp'),scope:'search-panel'},
  {key:'Alt-Enter',run:selectMatches,scope:'search-panel'},
  {key:'Ctrl-Enter',run:view=>{if(panels.get(view)?.replacementOpen)replaceAll(view);return true;},scope:'search-panel'},
  ...searchKeymap,indentWithTab,
]))];
