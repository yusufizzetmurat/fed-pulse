import { describe, expect, it } from "vitest";

import { toCsv } from "@/lib/export/csv";

describe("toCsv", () => {
  it("emits CRLF-delimited rows with comma separators", () => {
    const out = toCsv([
      ["field", "value"],
      ["symbol", "^GSPC"],
      ["close", 5500],
    ]);
    expect(out).toBe("field,value\r\nsymbol,^GSPC\r\nclose,5500");
  });

  it("quotes values containing commas, newlines, or quotes", () => {
    const out = toCsv([
      ["field", "value"],
      ["note", "hello, world"],
      ["multiline", "a\nb"],
      ["quoted", 'she said "hi"'],
    ]);
    expect(out).toBe(
      'field,value\r\nnote,"hello, world"\r\nmultiline,"a\nb"\r\nquoted,"she said ""hi"""',
    );
  });

  it("renders null/undefined as empty cells", () => {
    expect(toCsv([["a", null, undefined, ""]])).toBe("a,,,");
  });

  it("renders booleans and numbers without quoting", () => {
    expect(toCsv([["a", true, 0.5, -3]])).toBe("a,true,0.5,-3");
  });
});
