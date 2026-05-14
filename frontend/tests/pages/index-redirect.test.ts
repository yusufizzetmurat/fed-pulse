import { describe, expect, it } from "vitest";

import { getServerSideProps } from "@/pages/index";

describe("pages/index getServerSideProps", () => {
  it("redirects / to /analyze", async () => {
    const result = await getServerSideProps({} as Parameters<typeof getServerSideProps>[0]);
    expect(result).toEqual({
      redirect: { destination: "/analyze", permanent: false },
    });
  });
});
