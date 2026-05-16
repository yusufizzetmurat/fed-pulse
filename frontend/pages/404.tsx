import { ErrorPage } from "@/components/shell/error-page";

export default function NotFoundPage() {
  return (
    <ErrorPage
      status="404"
      title="That page is off the map"
      description="The URL you opened does not match any route on this Fed Pulse build. Common causes are stale bookmarks, copy-paste typos, or a feature that was renamed."
      hint={
        <>
          Try the <strong>Analyze</strong> tab to start a new run, or jump to
          {" "}
          <strong>History</strong> for prior analyses.
        </>
      }
    />
  );
}
