import { describe, expect, it } from "vitest";

import { categorizeError, errorMessage } from "@/lib/analyze/errors";

describe("categorizeError", () => {
  it("maps 400 to bad_input and surfaces the detail copy", () => {
    const result = categorizeError({
      response: { status: 400, data: { detail: "Symbol missing" } },
    });
    expect(result.category).toBe("bad_input");
    expect(result.message).toBe("Symbol missing");
  });

  it("maps 422 to bad_input with a default message when detail is empty", () => {
    const result = categorizeError({ response: { status: 422 } });
    expect(result.category).toBe("bad_input");
    expect(result.message).toMatch(/check the inputs/i);
  });

  it("maps 404 to not_found and surfaces the detail copy", () => {
    const result = categorizeError({
      response: { status: 404, data: { detail: "Run not in store" } },
    });
    expect(result.category).toBe("not_found");
    expect(result.message).toBe("Run not in store");
  });

  it("falls back to 'Not found.' when 404 has no detail", () => {
    const result = categorizeError({ response: { status: 404 } });
    expect(result.category).toBe("not_found");
    expect(result.message).toBe("Not found.");
  });

  it.each([500, 502, 503, 504])("maps %s to model_unavailable", (status) => {
    const result = categorizeError({ response: { status } });
    expect(result.category).toBe("model_unavailable");
    expect(result.message).toMatch(/model unavailable/i);
  });

  it("maps 409 to model_unavailable", () => {
    const result = categorizeError({ response: { status: 409 } });
    expect(result.category).toBe("model_unavailable");
  });

  it("maps code=ERR_NETWORK to network", () => {
    const result = categorizeError({ code: "ERR_NETWORK" });
    expect(result.category).toBe("network");
    expect(result.message).toMatch(/network error/i);
  });

  it("maps code=ETIMEDOUT to network", () => {
    const result = categorizeError({ code: "ETIMEDOUT" });
    expect(result.category).toBe("network");
  });

  it("treats a request without a response as network", () => {
    const result = categorizeError({ request: {} });
    expect(result.category).toBe("network");
  });

  it("treats 'Failed to fetch' message without response as network", () => {
    const result = categorizeError(new Error("Failed to fetch"));
    expect(result.category).toBe("network");
  });

  it("falls through to model_unavailable for a plain Error with no shape", () => {
    const result = categorizeError(new Error("boom"));
    expect(result.category).toBe("model_unavailable");
  });

  it("returns network + fallback copy for null", () => {
    const result = categorizeError(null);
    expect(result.category).toBe("network");
    expect(result.message).toMatch(/something went wrong/i);
  });

  it("returns network + fallback copy for undefined", () => {
    const result = categorizeError(undefined);
    expect(result.category).toBe("network");
    expect(result.message).toMatch(/something went wrong/i);
  });
});

describe("errorMessage", () => {
  it("returns the categorized message when err is present", () => {
    const message = errorMessage({
      response: { status: 400, data: { detail: "Bad date" } },
    });
    expect(message).toBe("Bad date");
  });

  it("returns the fallback when err is null and fallback is passed", () => {
    expect(errorMessage(null, "Could not load.")).toBe("Could not load.");
  });

  it("returns the default fallback copy when err is null and no fallback", () => {
    expect(errorMessage(null)).toMatch(/something went wrong/i);
  });
});
