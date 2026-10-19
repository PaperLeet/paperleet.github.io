import {FileCode2,FlaskConical,MessageCircle} from 'lucide-react';
export type WorkspacePanelIconProps={name:'file'|'lab'|'discussion';size?:number};
/** Decorative panel icons load with practice; their native control labels remain immediate. */
export default function WorkspacePanelIcon({name,size=14}:WorkspacePanelIconProps){
 const Icon={file:FileCode2,lab:FlaskConical,discussion:MessageCircle}[name];
 return <Icon size={size} aria-hidden="true"/>;
}
