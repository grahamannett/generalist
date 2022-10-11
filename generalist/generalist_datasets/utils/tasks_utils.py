from typing import Any, List


class TextSummaryMixin:
    default = "summary: "

    # def __call__(self, text: str, summary: str, title: str = None) -> str:
    #     base = f"{self.default}\n"
    #     if title:
    #         base += f"{title}\n"
    #     base += f"{text}\n"
    #     base += f"summary:\n{summary}"

    #     return base

    def __call__(self, summary: str):
        return f"{self.default}{summary}"


class ImageCaptioningMixin:
    default = "image caption: "

    def __call__(self, data: str, **kwds: Any) -> str:
        return f"{self.default}{data}"


class SegmentationMixin:
    default = "segmentation: "

    def __call__(self, data: str, **kwds: Any) -> str:
        return f"{self.default}{data}"


class RegionToCategoryMixin:
    default = "what is in this region: "

    def __call__(self, region: List[int], category: str, **kwds: Any) -> str:
        return f"{self.default}{' '.join([str(b) for b in region])}  {category}"


class TaskInterface:
    caption = ImageCaptioningMixin()
    categorize_region = RegionToCategoryMixin()
    text_summary = TextSummaryMixin()
