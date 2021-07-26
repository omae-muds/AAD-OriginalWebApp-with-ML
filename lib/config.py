from pydantic import BaseSettings


class Dotenv(BaseSettings):
    deta_project_key: str
    deta_project_id: str

    class Config:
        env_file = ".env"


class Settings(BaseSettings):
    deta_drive: str = "aad"
    deta_drive_txt_prefix: str = "txt/"
    deta_drive_img_prefix: str = "img/"

    deta_base: str = "aad"
    deta_base_features: str = "aad-features"
