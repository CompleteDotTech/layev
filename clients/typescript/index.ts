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
export class KevLayaClient {
  constructor(private readonly options: { baseURL: string; model?: string; apiKey?: string; timeoutMs?: number }) {}
  private async request<T>(path: string, body?: unknown): Promise<T> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.options.apiKey) headers.Authorization = `Bearer ${this.options.apiKey}`;
    const response = await fetch(this.options.baseURL.replace(/\/$/, "") + path, {
      method: body === undefined ? "GET" : "POST", headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(this.options.timeoutMs ?? 120000),
    });
    if (!response.ok) throw new Error(`Kev-Laya HTTP ${response.status}: ${await response.text()}`);
    return await response.json() as T;
  }
  systemOne(state: State, questions: Record<string, Question>, model?: string): Promise<SystemOneResponse> {
    return this.request("/v1/systemone", { state, questions, model: model ?? this.options.model ?? "kev-laya-preview" });
  }
  models(): Promise<{ models: Array<{ name: string; description: string; release_date: string }> }> {
    return this.request("/v1/models");
  }
}
