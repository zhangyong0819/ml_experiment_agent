FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 ENABLE_REAL_LSF=false ML_AGENT_ROOT=/app
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir .
COPY configs ./configs
COPY project_specs ./project_specs
COPY toy_project ./toy_project
COPY knowledge ./knowledge
RUN mkdir -p /app/data
VOLUME ["/app/data"]
EXPOSE 8000
CMD ["uvicorn", "ml_experiment_agent.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
