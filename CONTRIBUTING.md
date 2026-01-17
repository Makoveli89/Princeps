# Contributing to Princeps Brain & Console

## Getting Started

1. **Clone the repo**
2. **Install Backend Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install Frontend Dependencies**
   ```bash
   cd apps/console
   npm install
   ```

## Development

- **Run Full Stack**: `docker-compose up`
- **Run Backend Only**: `make run`
- **Run Frontend Only**: `cd apps/console && npm run dev`

## Code Standards

### Backend
- **Linting**: Ruff (`make lint`)
- **Formatting**: Black (`make format`)
- **Testing**: Pytest (`make test`)

### Frontend
- **Linting**: ESLint (`npm run lint`)
- **Formatting**: Prettier (`npm run format`)
- **Strict Mode**: TypeScript strict mode is enabled.

## Pull Requests
- Use the provided PR template.
- Ensure all CI checks pass before merging.
