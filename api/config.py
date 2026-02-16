"""Flask API configuration."""
import os


class Config:
    DEBUG = False
    TESTING = False
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class TestingConfig(Config):
    TESTING = True


config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}
