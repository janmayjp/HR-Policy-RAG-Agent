from app.core.config import get_settings
settings = get_settings()

print(f"App Name: {settings.app_name}")
print(f"openrouter_api_key: {settings.openrouter_api_key}")