import configparser


class Config:
    config = configparser.ConfigParser()
    config.read('configs/sd-config.ini', encoding="utf-8")
    LOG_FILE = config['DEFAULT']['log_file']
    CKPT_PATH = config['DEFAULT']['ckpt_path']
    CONFIG_FILE = config['DEFAULT']['config_file']
    OUTPUT_DIR: str = config['DEFAULT']['output_dir']
    TXT2IMG_OUT_DIR = OUTPUT_DIR + '/txt2img'
    DEVICE = config['DEFAULT']['device']
