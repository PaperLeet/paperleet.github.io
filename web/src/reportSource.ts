import type { Report } from "./types";

export type SourceFingerprint = { source: string; sha256?: string };
export type ReportSourceState = "none" | "unknown" | "checking" | "current" | "changed";

/** Match exact UTF-8 source bytes, including whitespace and line endings. */
export async function fingerprintSource(source: string): Promise<SourceFingerprint> {
  try {
    const digest = await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(source));
    return { source, sha256: Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("") };
  } catch {
    // Old/insecure browsers can still judge; they must not claim a version match.
    return { source };
  }
}

export function reportSourceState(report: Report | undefined, source: string, fingerprint?: SourceFingerprint): ReportSourceState {
  if (!report) return "none";
  if (!report.source_sha256) return "unknown";
  if (!fingerprint || fingerprint.source !== source) return "checking";
  if (!fingerprint.sha256) return "unknown";
  return fingerprint.sha256 === report.source_sha256 ? "current" : "changed";
}
