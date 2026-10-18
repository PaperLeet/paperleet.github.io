/** Route-dependent behavior is installed before mount, outside the initial app bundle. */
export type CatalogRoute = {id: string; number: number};
export type PathRouter = {
  read(value: string | URL): string;
  href(value: string | undefined): string | undefined;
  navigate(value: string): void;
  register(problems: CatalogRoute[]): void;
  sameUrl(url: URL): boolean;
};
let router: PathRouter | undefined;
export const pathRouting = () => typeof document !== 'undefined' && document.documentElement.dataset.routing === 'path';
export const installPathRouter = (value: PathRouter) => {router=value;};
export const publicHref = (href: string | undefined) => router ? router.href(href) : href;
export const routeHash = (value: string | URL) => router ? router.read(value) : new URL(value,location.href).hash;
export const sameAppUrl = (url: URL) => router ? router.sameUrl(url) : url.origin===location.origin&&url.pathname===location.pathname&&url.search===location.search;
export const registerCatalogRoutes = (problems: CatalogRoute[]) => router?.register(problems);
export const appLocation = {
  get hash() {return router ? router.read(location.href) : location.hash;},
  set hash(value: string) {if(router)router.navigate(value);else location.hash=value;},
};
