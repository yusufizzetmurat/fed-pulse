import { ErrorPage } from "@/components/shell/error-page";

export default function ServerErrorPage() {
  return (
    <ErrorPage
      status="500"
      title="Something broke on our side"
      description="The Fed Pulse server hit an unhandled error while building this page. The backend logs will carry the stack trace; reload after a moment or head back to a known-good route."
      hint={
        <>
          If the error sticks, confirm the backend is reachable at the URL exposed by
          {" "}
          <code className="font-mono text-foreground">NEXT_PUBLIC_API_URL</code>.
        </>
      }
    />
  );
}
