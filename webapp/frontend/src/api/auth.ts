import { api } from "./client";
import type { TokenResponse, UserResponse } from "./types";

export const authApi = {
  register: (email: string, password: string) =>
    api.post("/auth/register", { email, password }),

  login: async (email: string, password: string): Promise<string> => {
    const { data } = await api.post<TokenResponse>("/auth/login", { email, password });
    return data.access_token;
  },

  me: () => api.get<UserResponse>("/auth/me"),
};
