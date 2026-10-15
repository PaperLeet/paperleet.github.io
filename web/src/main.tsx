import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./style.css";
import "./ui-refinements.css";
import "./product-ui.css";
import "./workspace-polish.css";
import "./feedback-polish.css";
import "./library-workbench.css";
import { startPublicNavigation } from "./deployment";
import { AppErrorBoundary } from "./AppErrorBoundary";
import { finishAppStartup } from "./appRecovery";

function StartupReady() {
  React.useEffect(finishAppStartup, []);
  return null;
}
if (startPublicNavigation()) ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <StartupReady />
    <AppErrorBoundary><App /></AppErrorBoundary>
  </React.StrictMode>,
);
