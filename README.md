# Princeps

## 🔮 Activation

The platform is now fully configured for deployment with a sidecar architecture (Python Backend + React Frontend) and connected to your **Supabase** database with **Multi-Model Agents** enabled.

### 🚀 Launch

To activate the platform:

```bash
./start.sh
```

This will:
1.  Initialize the Python environment and install dependencies.
2.  Install frontend dependencies.
3.  Launch the **Backend API** on `http://localhost:8000`.
4.  Launch the **Princeps Console** on `http://localhost:5173`.

### 🔑 Configuration

Your environment is pre-configured in the `.env` file with:
*   **Database:** Supabase (Transaction Pooler Mode on port 6543)
*   **LLMs:** Anthropic (Claude), OpenAI, and Google (Gemini)

> **Note:** The database connection is configured to use the Supabase **Transaction Pooler** (port 6543) which is IPv4 compatible. If you encounter issues, ensure your network allows outbound traffic to port 6543.

### 🛠️ Troubleshooting

*   **Database Connection:** If you see "Database initialization failed", check `backend.log` for details.
*   **Port Conflicts:** Ensure ports `8000` and `5173` are free.
*   **Rebuild Frontend:** If you make UI changes, the development server (`start.sh`) handles hot-reloading. For production builds, run `cd apps/console && npm run build`.
