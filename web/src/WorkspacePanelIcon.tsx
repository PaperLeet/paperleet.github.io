import {ArrowDownToLine,FileCode2,FlaskConical,LockKeyhole,Maximize2,MessageCircle,Minimize2,Terminal} from 'lucide-react';
export type WorkspacePanelIconProps={name:'file'|'lab'|'discussion'|'focus'|'unfocus'|'download'|'lock'|'terminal';size?:number};
/** Decorative panel icons load with practice; their native control labels remain immediate. */
export default function WorkspacePanelIcon({name,size=14}:WorkspacePanelIconProps){
 const Icon={file:FileCode2,lab:FlaskConical,discussion:MessageCircle,focus:Maximize2,unfocus:Minimize2,download:ArrowDownToLine,lock:LockKeyhole,terminal:Terminal}[name];
 return <Icon size={size} aria-hidden="true"/>;
}
