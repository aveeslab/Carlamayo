import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if importlib.util.find_spec("alpamayo1_5") is None:
    alpamayo_package = types.ModuleType("alpamayo1_5")
    alpamayo_package.__path__ = []
    models_package = types.ModuleType("alpamayo1_5.models")
    models_package.__path__ = []
    model_module = types.ModuleType("alpamayo1_5.models.alpamayo1_5")
    config_module = types.ModuleType("alpamayo1_5.config")

    class Alpamayo1_5:
        pass

    class Alpamayo1_5Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    helper_module = types.ModuleType("alpamayo1_5.helper")
    helper_module.create_message = lambda *args, **kwargs: []
    helper_module.create_vqa_message = lambda *args, **kwargs: []
    helper_module.get_processor = lambda tokenizer: None
    helper_module.to_device = lambda data, *args, **kwargs: data

    model_module.Alpamayo1_5 = Alpamayo1_5
    config_module.Alpamayo1_5Config = Alpamayo1_5Config
    alpamayo_package.helper = helper_module
    sys.modules["alpamayo1_5"] = alpamayo_package
    sys.modules["alpamayo1_5.helper"] = helper_module
    sys.modules["alpamayo1_5.config"] = config_module
    sys.modules["alpamayo1_5.models"] = models_package
    sys.modules["alpamayo1_5.models.alpamayo1_5"] = model_module
