from pydantic import BaseModel


class Txt2imgOption(BaseModel):
    prompt: str = ''
    steps: int = 50
    plms: bool = False
    dpm: bool = False
    fixed_code: bool = False
    ddim_eta: float = 0.0
    n_iter: int = 1
    H: int = 512
    W: int = 512
    C: int = 4
    f: int = 8
    n_samples: int = 1
    n_rows: int = 1
    scale: float = 9.0
    from_file: str = None
    ckpt: str = None
    seed: int = None
    precision: str = 'full'
    repeat: int = 1
    torchscript: bool = False
    ipex: bool = False
    bf16: bool = False
