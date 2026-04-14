FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure runtime directories exist
RUN mkdir -p context

CMD ["python", "-m", "bot.bot"]
