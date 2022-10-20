from typing import Any, List


class TaskBaseClass:
    task_type: str = None
    default: str = None


class TextSummaryMixin(TaskBaseClass):
    task_type: str = "summary"
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

    # @property
    # def default(self):
    #     return self._default


class ImageCaptioningMixin(TaskBaseClass):
    task_type: str = "image captioning"
    default = "image caption: "

    def __call__(self, data: str, **kwds: Any) -> str:
        return f"{self.default}{data}"


class SegmentationMixin(TaskBaseClass):
    task_type: str = "segmentation"
    default = "segmentation: "

    def __call__(self, data: str, **kwds: Any) -> str:
        return f"{self.default}{data}"


class RegionToCategoryMixin(TaskBaseClass):
    task_type: str = "region to category"
    default = "what is in this region: "

    def __call__(self, region: List[int], category: str, **kwds: Any) -> str:
        return f"{self.default}{' '.join([str(b) for b in region])}  {category}"


class TaskInterface:
    caption = ImageCaptioningMixin()
    categorize_region = RegionToCategoryMixin()
    text_summary = TextSummaryMixin()
