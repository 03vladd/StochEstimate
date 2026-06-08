"""Quick smoke test — run from webapp/backend/ with the venv active."""
import os, sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv(".env")

from app.services.narration_service import generate_narration

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
print(f"API key present: {bool(api_key)} (length {len(api_key)})")

result = generate_narration(
    ticker1="AMZN", ticker2="WMT",
    confidence_level="MEDIUM", tests_passed=3,
    theta=0.0162, mu=1.34, sigma=4.48,
    theta_ci=(0.008, 0.024), sigma_ci=(4.05, 4.91),
    z_score=0.73, signal_type="NONE",
    api_key=api_key,
)

print("\n--- Result ---")
print(result)
