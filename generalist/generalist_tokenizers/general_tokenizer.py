class GeneralTokenizer:
    data_type = None
    _instance = None
    _singleton = False

    def __init__(self, device: str, **kwargs):
        self.device = device
        self.check_instance()

    def check_instance(self):
        if (self.__class__._instance is None) and (self.__class__._singleton is False):
            return
        raise Exception("not okay fam -> singleton related")

    def fix_device(self, prop: str) -> None:
        raise NotImplementedError()
