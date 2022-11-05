from typing import Any, List


class TaskBaseClass:
    task_type: str = None
    default: str = None


class TextSummaryMixin(TaskBaseClass):
    task_type: str = "summary"
    default = "summary: "
    passage = "passage: "

    def __call__(self, summary: str, document: str = None):
        summary = f"{self.default}{summary}"
        if document:
            document = f"{self.passage}{document}"
        return (summary, document) if document else summary


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
