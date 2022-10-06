from typing import Any


class ImageCaptioningMixin:
    default = "image caption: "

    def __call__(self, data: str, **kwds: Any) -> str:
        return f"{self.default}{data}"


class TaskInterface:
    caption = ImageCaptioningMixin()


# class CocoCaptionsMixin:
#     pass


# class CocoInstancesMixin:
#     pass
