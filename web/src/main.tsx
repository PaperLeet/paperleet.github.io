import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./style.css";
import "./ui-refinements.css";
import "./product-ui.css";
import "./workspace-polish.css";
import "./feedback-polish.css";
import "./header-account.css";
import "./library-workbench.css";
import "./control-surfaces.css";
import "./library-filters.css";
import "./visual-polish.css";
import { startPublicNavigation } from "./deployment";
import { AppErrorBoundary } from "./AppErrorBoundary";
import { finishAppStartup } from "./appRecovery";
import {pathRouting} from './appLocation';

function StartupReady() {
  React.useEffect(finishAppStartup, []);
  return null;
}
function mount() { if (startPublicNavigation()) ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <StartupReady />
    <AppErrorBoundary><App /></AppErrorBoundary>
  </React.StrictMode>,
);
}
if (pathRouting()) void import('./pathNavigation').then(({startPathNavigation})=>{startPathNavigation();mount();});
else mount();
