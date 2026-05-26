import { describe, expect, it, vi, beforeEach } from "vitest";
import axios from "axios";

import { postAnalyzeAnalogs } from "@/lib/analyze/api";

vi.mock("axios", () => ({
  default: { post: vi.fn() },
}));

const mockedAxios = axios as unknown as { post: ReturnType<typeof vi.fn> };

describe("postAnalyzeAnalogs", () => {
  beforeEach(() => {
    mockedAxios.post.mockReset();
  });

  it("posts the request payload to /analyze/analogs and returns parsed analogs", async () => {
    mockedAxios.post.mockResolvedValue({
      data: {
        analogs: [
          {
            event_date: "2019-07-31",
            similarity: 0.82,
            axis_stance: "dovish",
            subsequent_vol_regime: "high",
            excerpt: "Information received…",
          },
        ],
        index_size: 184,
        encoder_alias: "finbert_fed_adjacent_xbank_dapt_retrieval",
      },
    });
    const result = await postAnalyzeAnalogs("http://api", {
      text: "Recent indicators…",
      k: 5,
      as_of_date: "2026-04-30",
    });
    expect(mockedAxios.post).toHaveBeenCalledWith(
      "http://api/analyze/analogs",
      { text: "Recent indicators…", k: 5, as_of_date: "2026-04-30" },
      { signal: undefined },
    );
    expect(result.analogs).toHaveLength(1);
    expect(result.index_size).toBe(184);
    expect(result.encoder_alias).toBe("finbert_fed_adjacent_xbank_dapt_retrieval");
  });

  it("returns a bundle-absent shape when the response body is missing", async () => {
    mockedAxios.post.mockResolvedValue({ data: undefined });
    const result = await postAnalyzeAnalogs("http://api", { text: "x" });
    expect(result.analogs).toEqual([]);
    expect(result.index_size).toBe(0);
    expect(result.encoder_alias).toBe("");
  });

  it("forwards the abort signal", async () => {
    mockedAxios.post.mockResolvedValue({
      data: { analogs: [], index_size: 0, encoder_alias: "" },
    });
    const controller = new AbortController();
    await postAnalyzeAnalogs("http://api", { text: "x" }, controller.signal);
    expect(mockedAxios.post).toHaveBeenCalledWith(
      "http://api/analyze/analogs",
      { text: "x" },
      { signal: controller.signal },
    );
  });
});
