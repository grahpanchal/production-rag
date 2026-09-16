from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    groq_api_key: str

    api_key: str

    qdrant_url: str = "http://localhost:6333"

    qdrant_collection: str = "documents"

    cors_origins: str = "http://localhost:3000," "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()

# from pydantic_settings import BaseSettings, SettingsConfigDict


# class Settings(BaseSettings):

#     groq_api_key: str

#     api_key: str

#     qdrant_url: str = "http://localhost:6333"

#     qdrant_collection: str = "documents"

#     model_config = SettingsConfigDict(env_file=".env")


# settings = Settings()
