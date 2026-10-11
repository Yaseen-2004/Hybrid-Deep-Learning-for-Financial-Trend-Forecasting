# Security and operational notes

- Do not commit `.env`, `.streamlit/secrets.toml`, API keys, or user data.
- The upload workflow analyzes files in memory and does not save the raw upload to the project folders.
- The dashboard has no authentication/authorization layer. Place it behind trusted access controls before any public deployment, and review whether uploaded files may contain sensitive data.
- Docker runs as an unprivileged user; model/result/data folders are mounted read-only by the included Compose configuration.
- Treat `.keras`, `.joblib`, and JSON config files as trusted artifacts. Only load artifacts from a controlled build/training process. Never accept user-uploaded model binaries for loading.
- The project is educational, not investment advice or a trading service. Public or operational financial use needs independent security, legal, data-license, and model-risk review.
