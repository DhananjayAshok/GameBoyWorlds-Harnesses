from utils.parameter_handling import load_parameters
from utils.log_handling import log_warn, log_error
from typing import Any, List, Union
import numpy as np
from abc import ABC
from PIL import Image
from utils.lm_inference import (
    OpenAIModel,
    OpenRouterModel,
    GatewayModel,
    AnthropicModel,
    vLLMModel,
    parse_yes_no,
)


def convert_numpy_greyscale_to_pillow(arr: np.ndarray) -> Image:
    if len(arr.shape) == 2:
        arr = arr.reshape(arr.shape[0], arr.shape[1], 1)
    rgb = np.stack([arr[:, :, 0], arr[:, :, 0], arr[:, :, 0]], axis=2)
    return Image.fromarray(rgb)


def get_converted_image_list(nested_image_list) -> list[Image.Image]:
    finals = []
    for item in nested_image_list:
        if isinstance(item, list):
            finals.append(get_converted_image_list(item))
        elif isinstance(item, Image.Image):
            finals.append(item)
        elif isinstance(item, np.ndarray):
            finals.append(convert_numpy_greyscale_to_pillow(item))
        else:
            log_error(
                f"Invalid image format: {type(item)}. Expected PIL Image or numpy array."
            )
    return finals


class VLM:
    def __init__(self, model_name, vlm_kind):
        self._model_name = model_name
        self._vlm_kind = vlm_kind
        self._vlm = None
        if self._vlm_kind == "openai":
            self._vlm = OpenAIModel(model=self._model_name)
        elif self._vlm_kind == "openrouter":
            self._vlm = OpenRouterModel(model=self._model_name)
        elif self._vlm_kind == "gateway":
            self._vlm = GatewayModel(model=self._model_name)
        elif self._vlm_kind == "huggingface":
            from utils.huggingface_inference import HuggingFaceModel

            self._vlm = HuggingFaceModel(model=self._model_name, model_kind="vlm")
        elif self._vlm_kind == "anthropic":
            self._vlm = AnthropicModel(model=self._model_name)
        elif self._vlm_kind == "vllm":
            self._vlm = vLLMModel(model=self._model_name)
        else:
            raise ValueError(f"Invalid VLM kind: {self._vlm_kind}")

    def infer(
        self,
        texts: Union[str, list[str]],
        max_new_tokens: int,
        images: Union[
            list[Union[Image.Image, np.ndarray]],
            list[list[Union[Image.Image, np.ndarray]]],
        ] = None,
        temperature: float = None,
        n_outputs: int = 1,
    ) -> dict[str, Any]:
        if images is not None:
            treated_images = get_converted_image_list(images)
            images = treated_images
        return self._vlm.infer(
            texts=texts,
            images=images,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=n_outputs,
        )


class NamedVLM(VLM, ABC):
    NAME = None

    def __init__(self, parameters: dict = None):
        parameters = load_parameters(parameters)
        model_name = parameters[f"{self.NAME}_vlm_model"]
        vlm_kind = parameters[f"{self.NAME}_vlm_kind"]
        super().__init__(model_name=model_name, vlm_kind=vlm_kind)


class ExecutorVLM(NamedVLM):
    NAME = "executor"


class ObjectDetectionVLM(NamedVLM):
    NAME = "object_detection"


def merge_ocr_strings(strings, min_overlap=3):
    current_strings = list(set(s.strip() for s in strings if s.strip()))

    current_strings.sort(key=len, reverse=True)
    final_set = []
    for s in current_strings:
        if not any(s in other for other in final_set):
            final_set.append(s)

    merged_list = final_set[:]
    changed = True

    while changed:
        changed = False
        i = 0
        while i < len(merged_list):
            j = 0
            while j < len(merged_list):
                if i == j:
                    j += 1
                    continue

                s1, s2 = merged_list[i], merged_list[j]

                overlap_len = 0
                max_possible_overlap = min(len(s1), len(s2))

                for length in range(max_possible_overlap, min_overlap - 1, -1):
                    if s1.endswith(s2[:length]):
                        overlap_len = length
                        break

                if overlap_len > 0:
                    new_string = s1 + s2[overlap_len:]

                    val_i = merged_list[i]
                    val_j = merged_list[j]
                    merged_list.remove(val_i)
                    merged_list.remove(val_j)
                    merged_list.append(new_string)

                    changed = True
                    i = -1
                    break
                j += 1
            if changed:
                break
            i += 1

    return merged_list


DEFAULT_OCR_PROMPT = "If there is no text in the image, just say NONE. Otherwise, perform OCR and state the text in this image:"
OCR_CURSOR_MARGIN = 12
OCR_BACKGROUND = 248
_OCR_CACHE: dict[str, str] = {}


def ocr_crop_key(crop: np.ndarray) -> str | None:
    from execution.perception.matching import cell_key

    if crop is None or not hasattr(crop, "shape") or crop.size == 0:
        return None
    body = crop[:, :-OCR_CURSOR_MARGIN] if crop.shape[1] > OCR_CURSOR_MARGIN else crop
    mask = np.ascontiguousarray(body >= OCR_BACKGROUND, dtype=np.uint8)
    return cell_key(mask)


def clean_ocr_output(text: str) -> str:
    text = str(text or "")
    text = text.rsplit("Response:", 1)[1] if "Response:" in text else text
    text = text.strip()
    for lead in ("The text in the image is:", "The text in this image is:"):
        if text.startswith(lead):
            text = text[len(lead):].strip()
    return text.strip("*").strip()


def cache_lookup_ocr(crop: np.ndarray) -> str | None:
    key = ocr_crop_key(crop)
    return _OCR_CACHE.get(key) if key is not None else None


def cache_store_ocr(crop: np.ndarray, text: str) -> str:
    cleaned = clean_ocr_output(text)
    key = ocr_crop_key(crop)
    if key is not None:
        _OCR_CACHE[key] = cleaned
    return cleaned


def ocr(
    images: List[np.ndarray],
    vlm: VLM,
    *,
    text_prompt=None,
    do_merge: bool = True,
    max_new_tokens: int = 4000,
    parameters: dict = None,
) -> List[str]:
    parameters = load_parameters(parameters)
    if not images:
        return []
    use_cache = text_prompt is None or text_prompt == DEFAULT_OCR_PROMPT
    if text_prompt is None:
        text_prompt = DEFAULT_OCR_PROMPT
    if vlm is None:
        log_error("ocr() needs a vlm to be passed in.", parameters)

    results: List[str | None] = [cache_lookup_ocr(img) if use_cache else None for img in images]
    missing_indices = [i for i, res in enumerate(results) if res is None]
    if missing_indices:
        missing_images = [[images[i]] for i in missing_indices]
        raw_out = vlm.infer(
            texts=[text_prompt] * len(missing_images),
            images=missing_images,
            max_new_tokens=max_new_tokens,
        )["output"]
        if not isinstance(raw_out, list):
            raw_out = [raw_out]
        for idx, raw in zip(missing_indices, raw_out):
            cleaned = cache_store_ocr(images[idx], raw) if use_cache else clean_ocr_output(raw)
            results[idx] = cleaned
            if cleaned.lower() == "none":
                img = images[idx]
                log_warn(
                    f"Got NONE as output from OCR. Could this have been avoided?\nimages statistics: Max: {img.max()}, Min: {img.min()}, Mean: {img.mean()}, percentage of non zero cells {(img > 0).mean()}, percentage of non 255 cells {(img < 255).mean()}",
                    parameters,
                )

    ocred = [res.strip() for res in results if res and res.strip().lower() != "none"]
    if do_merge:
        ocred = merge_ocr_strings(ocred)
    return ocred


def object_detection(
    description: str,
    images: List[np.ndarray],
    text_prompt: str = None,
    model: VLM = None,
    parameters: dict = None,
) -> List[bool]:
    """A caller-supplied text_prompt must request an `Answer: <yes or no>` line."""
    if text_prompt is None:
        text_prompt = f"""You are playing a gameboy game and are given a screen capture of the game.
        Your job is to locate the target that best fits the description `{description}`

        Do you see the target described? Answer in the following format:
        Explanation: <a single sentence describing what you see>
        Answer: <yes or no>[STOP]"""
    if model is None:
        model = ObjectDetectionVLM(parameters=parameters)
    outputs = model.infer(
        texts=[text_prompt for _ in images],
        images=[[image] for image in images],
        max_new_tokens=60,
    )["output"]
    founds = []
    for i, output in enumerate(outputs):
        verdict = parse_yes_no(output, "Answer")
        if verdict is None:
            log_warn(
                f"Object detection for {description!r} gave no parseable Answer line; "
                f"treating as not detected. Raw output: {output!r}",
                parameters,
            )
        founds.append(verdict is True)
    return founds


def identify_matches(
    description: str,
    screens: List[np.ndarray],
    reference: np.ndarray,
    text_prompt: str = None,
    model: VLM = None,
    parameters: dict = None,
) -> List[bool]:
    """A caller-supplied text_prompt must request an `Answer: <yes or no>` line."""
    reference = convert_numpy_greyscale_to_pillow(reference)
    if text_prompt is None:
        text_prompt = f"The target, described as {description} is shown as reference in Picture 1. Does Picture 2 contain the object from Picture 1 in it? Answer in the following format: \nExplanation: <briefly describe what is in Picture 2, with reference to the image in Picture 1>\nAnswer: <Yes or No>[STOP]"
    texts = [text_prompt for _ in screens]
    images = []
    for screen in screens:
        images.append([reference, screen])
    if model is None:
        model = ObjectDetectionVLM(parameters=parameters)
    outputs = model.infer(texts=texts, images=images, max_new_tokens=120)["output"]
    results = []
    for output in outputs:
        verdict = parse_yes_no(output, "Answer")
        if verdict is None:
            log_warn(
                f"Match identification for {description!r} gave no parseable Answer line; "
                f"treating as no match. Raw output: {output!r}",
                parameters,
            )
        results.append(verdict is True)
    return results
