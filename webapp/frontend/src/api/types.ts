export interface UserResponse {
  id: string;
  email: string;
  is_verified: boolean;
  plan: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface PairSummary {
  pair_id: string;
  ticker1: string;
  ticker2: string;
  sector: string | null;
  confidence_level: string | null;
  latest_z_score: number | null;
  latest_signal_type: string | null;
  latest_signal_date: string | null;
  alert_enabled: boolean;
}

export interface CointegrationData {
  hedge_ratio: number;
  adf_statistic: number;
  adf_pvalue: number;
  cointegrated: boolean;
  computed_at: string;
}

export interface ValidationData {
  stationarity_passed: boolean;
  drift_passed: boolean;
  volatility_passed: boolean;
  acf_passed: boolean;
  normality_passed: boolean;
  confidence_level: string;
  tests_passed_count: number;
  acf_r_squared: number | null;
  acf_theta: number | null;
  computed_at: string;
}

export interface EstimationData {
  theta: number;
  mu: number;
  sigma: number;
  theta_ci_lower: number;
  theta_ci_upper: number;
  mu_ci_lower: number;
  mu_ci_upper: number;
  sigma_ci_lower: number;
  sigma_ci_upper: number;
  theta_std: number;
  sigma_std: number;
  n_mc_samples: number;
  model_version: string;
  computed_at: string;
}

export interface MLEData {
  theta: number;
  mu: number;
  sigma: number;
  log_likelihood: number | null;
  success: boolean;
  computed_at: string;
}

export interface SignalPoint {
  date: string;
  z_score: number;
  signal_type: "LONG" | "SHORT" | "EXIT" | "NONE";
  spread_value: number;
  stationary_mean: number;
  stationary_std: number;
}

export interface UserPairSettings {
  alert_enabled: boolean;
  long_threshold: number;
  short_threshold: number;
}

export interface PairDetail {
  pair_id: string;
  ticker1: string;
  ticker2: string;
  sector: string | null;
  narration: string | null;
  window_months: number | null;
  settings: UserPairSettings;
  cointegration: CointegrationData | null;
  validation: ValidationData | null;
  estimation: EstimationData | null;
  mle: MLEData | null;
  signals: SignalPoint[];
}

export interface JobStatus {
  job_id: string;
  status: "pending" | "running" | "complete" | "failed";
  created_at: string;
  completed_at: string | null;
  error_message: string | null;
}

export interface PairSubmitResponse {
  pair_id: string;
  job_id: string;
  status: "pending" | "cached";
}
