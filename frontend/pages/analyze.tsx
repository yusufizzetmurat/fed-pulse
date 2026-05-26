import type { GetServerSideProps } from "next";

export const getServerSideProps: GetServerSideProps = async ({ query }) => {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (typeof value === "string") {
      params.set(key, value);
    } else if (Array.isArray(value) && typeof value[0] === "string") {
      params.set(key, value[0]);
    }
  }
  const suffix = params.toString();
  return {
    redirect: {
      destination: suffix ? `/?${suffix}` : "/",
      permanent: false,
    },
  };
};

export default function AnalyzeRedirect() {
  return null;
}
