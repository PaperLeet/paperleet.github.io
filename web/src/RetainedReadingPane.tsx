import {Component,createRef,type ReactNode} from 'react';

type Props={active:boolean;id:string;labelledBy:string;children:ReactNode};
type Position={top:number;left:number};

/** Mount content on first visit; snapshot before hiding so a collapsed scroller cannot erase its position. */
export default class RetainedReadingPane extends Component<Props,{visited:boolean},Position|null>{
 state={visited:this.props.active};
 private pane=createRef<HTMLDivElement>();
 private position:Position={top:0,left:0};
 static getDerivedStateFromProps(props:Props,state:{visited:boolean}){return props.active&&!state.visited?{visited:true}:null;}
 getSnapshotBeforeUpdate(previous:Props){const pane=this.pane.current;return previous.active&&!this.props.active&&pane?{top:pane.scrollTop,left:pane.scrollLeft}:null;}
 componentDidUpdate(previous:Props,_state:{visited:boolean},snapshot:Position|null){
  if(snapshot)this.position=snapshot;
  if(this.props.active&&!previous.active&&this.pane.current){this.pane.current.scrollTop=this.position.top;this.pane.current.scrollLeft=this.position.left;}
 }
 render(){
  const {active,id,labelledBy,children}=this.props;
  return <div ref={this.pane} className="description-scroll" id={id} role="tabpanel" aria-labelledby={labelledBy} tabIndex={0} hidden={!active}>{(active||this.state.visited)&&children}</div>;
 }
}
