import type {ComponentProps} from 'react';
import {publicHref} from './appLocation';

/** Real hrefs also work for crawlers, copy-link, and opening a new tab. */
export default function AppLink(props: ComponentProps<'a'>) {
  return <a {...props} href={publicHref(props.href)} />;
}
