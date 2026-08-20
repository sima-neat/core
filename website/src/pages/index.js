import React from "react";
import {Redirect, useLocation} from "@docusaurus/router";
import useBaseUrl from "@docusaurus/useBaseUrl";

export default function Home() {
  const location = useLocation();
  return <Redirect to={`${useBaseUrl("/getting-started/")}${location.search}`} />;
}
