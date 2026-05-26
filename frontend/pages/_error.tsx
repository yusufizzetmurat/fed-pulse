import type { NextPage, NextPageContext } from "next";

import { ErrorPage } from "@/components/shell/error-page";

interface ErrorProps {
  statusCode: number;
}

const FallbackError: NextPage<ErrorProps> = ({ statusCode }) => {
  if (statusCode === 404) {
    return (
      <ErrorPage
        status="404"
        title="That page is off the map"
        description="The URL you opened does not match any route on this Fed Pulse build."
      />
    );
  }
  if (statusCode >= 500) {
    return (
      <ErrorPage
        status={String(statusCode)}
        title="Something broke on our side"
        description="The Fed Pulse server hit an unhandled error while building this page. Reload after a moment or head back to a known-good route."
      />
    );
  }
  return (
    <ErrorPage
      status={String(statusCode || "Error")}
      title="An unexpected error occurred"
      description="Fed Pulse caught an error it does not have a branded page for. Head back to Analyze and try again."
    />
  );
};

FallbackError.getInitialProps = async ({ res, err }: NextPageContext) => {
  const statusCode = res?.statusCode ?? err?.statusCode ?? 404;
  return { statusCode };
};

export default FallbackError;
