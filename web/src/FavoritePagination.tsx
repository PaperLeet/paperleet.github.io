import LibraryPagination from './LibraryPagination';
export type Props=Omit<Parameters<typeof LibraryPagination>[0],'modules'|'labels'>&{canAct:()=>boolean};
export default function FavoritePagination({canAct,...props}:Props){
 return <LibraryPagination {...props} canAct={canAct} onPage={page=>{if(canAct())props.onPage(page);}} onSize={size=>{if(canAct())props.onSize(size);}} modules labels={{nav:props.compact?['收藏快捷翻页','Quick saved problem pages']:['收藏分页','Saved problem pages'],jump:props.compact?['快捷跳转收藏页码','Quick jump to saved problem page']:['跳转收藏页码','Jump to saved problem page'],count:['道题','problems']}}/>;
}
