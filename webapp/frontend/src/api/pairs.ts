import { api } from "./client";
import type {
  JobStatus,
  PairDetail,
  PairSubmitResponse,
  PairSummary,
  UserPairSettings,
} from "./types";

export const pairsApi = {
  submit: (ticker1: string, ticker2: string, window_months = 12) =>
    api.post<PairSubmitResponse>("/pairs", { ticker1, ticker2, window_months }),

  list: () => api.get<PairSummary[]>("/pairs"),

  detail: (pairId: string) => api.get<PairDetail>(`/pairs/${pairId}`),

  jobStatus: (pairId: string) => api.get<JobStatus>(`/pairs/${pairId}/job`),

  updateSettings: (pairId: string, settings: Partial<UserPairSettings>) =>
    api.patch<UserPairSettings>(`/pairs/${pairId}/settings`, settings),

  remove: (pairId: string) => api.delete(`/pairs/${pairId}`),

  generateNarration: (pairId: string) =>
    api.post<{ status: string }>(`/pairs/${pairId}/narration`, {}),

  refresh: (pairId: string) =>
    api.post<import("./types").JobStatus>(`/pairs/${pairId}/refresh`, {}),
};
