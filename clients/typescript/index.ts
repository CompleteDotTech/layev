export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export type Entry = string | { [key: string]: Json } | Json[] | null;
export type State = Exclude<Entry, null>;
export type Question =
  | { type: "choice"; instructions: Entry; criteria: Record<string, Entry> }
  | { type: "score"; instructions: Entry; criteria: Entry[] }
  | { type: "noul"; instructions: Entry; criteria?: { true?: Entry; false?: Entry } | null };
export type Answer =
  | { type: "choice"; choice: string; confidence: number; probabilities: Record<string, number> }
  | { type: "score"; score: number; confidence: number; probabilities: Record<string, number>; legend: Record<string, string> }
  | { type: "noul"; noul: number };
export interface SystemOneResponse {
  model: string;
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number; forward_tokens: number; state_tokens: number; branch_tokens: number[]; generated_tokens: 0 };
  confidence_definition: "entropy-concentration-v1";
  calibration_status: string;
}
/** Client contract identity, not a claim of TypeSafe or model equivalence. */
export const CLIENT_CONTRACT_VERSION = "layev-client/1" as const;

export interface SystemOneRequest {
  state: State;
  questions: Record<string, Question>;
  model: string;
}
export interface ModelsResponse {
  models: Array<{ name: string; description: string; release_date: string }>;
}
export interface KevLayaClientOptions {
  baseURL: string;
  model?: string;
  apiKey?: string;
  timeoutMs?: number;
}

/** HTTP errors deliberately exclude server bodies, URLs and bearer credentials. */
export class KevLayaHTTPError extends Error {
  constructor(public readonly status: number) {
    super(`Kev-Laya HTTP ${status}`);
    this.name = "KevLayaHTTPError";
  }
}

export class KevLayaClient {
  private readonly options: Readonly<KevLayaClientOptions>;
  private readonly baseURL: string;
  private readonly timeoutMs: number;

  constructor(options: KevLayaClientOptions) {
    // Copy options: a caller must not be able to redirect an existing client's key
    // by mutating the object they passed to the constructor.
    const url = new URL(options.baseURL);
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
      throw new TypeError("baseURL must be an HTTP(S) URL without credentials, query or fragment");
    }
    this.timeoutMs = options.timeoutMs ?? 120000;
    if (!Number.isSafeInteger(this.timeoutMs) || this.timeoutMs < 1 || this.timeoutMs > 2147483647) {
      throw new RangeError("timeoutMs must be an integer from 1 to 2147483647");
    }
    this.options = Object.freeze({ ...options });
    this.baseURL = url.toString().replace(/\/+$/, "");
  }

  private async request<T>(path: string, body?: unknown): Promise<T> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.options.apiKey) headers.Authorization = `Bearer ${this.options.apiKey}`;
    const response = await fetch(this.baseURL + path, {
      method: body === undefined ? "GET" : "POST", headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeoutMs),
      // Do not forward a request or credential to a redirect target. No retries.
      redirect: "error",
    });
    if (!response.ok) {
      // Do not buffer or expose an untrusted error response (which may echo data).
      await response.body?.cancel().catch(() => undefined);
      throw new KevLayaHTTPError(response.status);
    }
    // Fetch's timeout signal covers body consumption too. Preserve its abort error.
    const text = await response.text();
    try {
      return JSON.parse(text) as T;
    } catch {
      throw new Error("Kev-Laya returned invalid JSON");
    }
  }

  systemOne(state: State, questions: Record<string, Question>, model?: string): Promise<SystemOneResponse> {
    const body: SystemOneRequest = {
      state, questions, model: model ?? this.options.model ?? "kev-laya-preview",
    };
    return this.request("/v1/systemone", body);
  }

  models(): Promise<ModelsResponse> {
    return this.request("/v1/models");
  }
}
