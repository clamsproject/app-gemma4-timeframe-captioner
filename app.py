"""
Gemma 4 Timeframe Captioner CLAMS app (prototype, experimental backend).

Captions video frames sampled from input TimeFrame annotations with a Gemma
4 vision-language model from a GGUF file (``ClamsGGUFPromptableApp``). The
``model`` parameter picks the size and quantization (see metadata.py).

Gemma's reasoning block (``<|channel>thought ... <channel|>``) is split off
the caption, and kept on it when ``useReasoning`` is set.
"""

import argparse
import logging
from collections import Counter
from typing import List

from clams import Restifier
from clams.app import ClamsGGUFPromptableApp, PromptTask
from mmif import Mmif, View, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh

from metadata import GEMMA4_FILES

_RAW_RESPONSE_LOG_TRUNCATE = 200
# Gemma 4's reasoning block, as llama-server returns it with
# ``--reasoning-format none``
_THOUGHT_OPEN = '<|channel>thought'
_THOUGHT_CLOSE = '<channel|>'


class Gemma4TimeframeCaptioner(ClamsGGUFPromptableApp):

    # the GGUF files of each model id in ``analyzer_versions``
    MODEL_FILES = GEMMA4_FILES
    # UBATCH: room for an image's full token budget (up to 1,120);
    # N_PARALLEL: one slot, as the SDK recommends
    LLAMA_ARGS = {'LLAMA_ARG_UBATCH': '2048', 'LLAMA_ARG_N_PARALLEL': '1'}

    def _appmetadata(self):
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        self.logger.debug(f"Annotating with parameters: {parameters}")

        # start (or reuse) the server of the chosen family member
        self.load_model(parameters['model'])

        prompt: List[str] = list(parameters['prompt'])
        system_prompt: str = parameters['systemPrompt']
        prompt_mode: str = parameters['promptMode']
        max_images: int = parameters['maxImagesPerPrompt']
        images_mode: str = parameters['imagesPerPromptMode']
        use_reasoning: bool = bool(parameters['useReasoning'])
        tflabels_of_interest: List[str] = list(parameters.get('tfLabels') or [])
        tflabels_to_ignore: List[str] = list(
            parameters.get('tfLabelIgnore') or [])
        gen_params = dict(
            max_new_tokens=parameters['maxNewTokens'],
            temperature=parameters['temperature'],
            top_p=parameters['topP'],
            top_k=parameters['topK'],
            # drives the template's enable_thinking switch
            use_reasoning=use_reasoning,
        )

        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)
        new_view.new_contain(AnnotationTypes.Alignment)

        if not mmif.get_documents_by_type(DocumentTypes.VideoDocument):
            raise ValueError(
                "Gemma 4 Captioner requires a VideoDocument input.")

        # one task per TimeFrame; images the video cannot provide are
        # already dropped (and warned about) by vdh
        tasks = [
            PromptTask(images=images, origins=tp_ids, source=tf.id)
            for tf, images, tp_ids in vdh.iter_timeframe_images(
                mmif, new_view, tflabels_of_interest, tflabels_to_ignore,
                as_PIL=True)]
        if not tasks:
            self.logger.warning(
                "No matching TimeFrames yielded any sampled frames; "
                "nothing to process.")
            return mmif

        # Composite captioning: each task's image list is sent to the model
        # in one prompt and yields one caption. A TF with more frames than
        # ``maxImagesPerPrompt`` is chunked into several tasks, so it gets
        # several captions, each grounded to its own TimePoints. The
        # server's slot count sets how many prompts run at once.
        tasks = self.chunk_tasks(tasks, max_images, images_mode)
        chunks_per_tf = Counter(task.source for task in tasks)
        texts: List[str] = self.generate_batched(
            prompt=prompt,
            system_prompt=system_prompt,
            images=[task.images for task in tasks],
            prompt_mode=prompt_mode,
            **gen_params)

        chunks_seen: Counter = Counter()
        for i, (task, raw) in enumerate(zip(tasks, texts)):
            chunks_seen[task.source] += 1
            task_desc = (
                f"task[{i}]: TF={task.source} "
                f"chunk {chunks_seen[task.source]}/"
                f"{chunks_per_tf[task.source]} "
                f"({len(task.images)} images)")
            text, trace = self.split_tagged_reasoning_trace(
                raw, open_tag=_THOUGHT_OPEN, close_tag=_THOUGHT_CLOSE)
            if raw.lstrip().startswith(_THOUGHT_OPEN) and trace is None:
                # opened but never closed: the reasoning overran
                # maxNewTokens, so there is no caption
                text, trace = '', raw.strip()
                self.logger.warning(
                    f"  {task_desc} produced no caption -- the reasoning "
                    f"trace overran maxNewTokens "
                    f"(={parameters['maxNewTokens']}) without terminating; "
                    f"storing an empty caption. Raise maxNewTokens.")
            truncated = repr(text)
            if len(truncated) > _RAW_RESPONSE_LOG_TRUNCATE:
                truncated = truncated[:_RAW_RESPONSE_LOG_TRUNCATE] + '...'
            self.logger.debug(f"  {task_desc} caption: {truncated}")
            self.response_to_grounded_textdocument(
                new_view, source=task.source, response=text,
                reasoning_trace=(trace if use_reasoning else None),
                origins=task.origins, origination='derived')

        return mmif


def get_app():
    return Gemma4TimeframeCaptioner()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000",
                        help="set port to listen")
    parser.add_argument("--production", action="store_true",
                        help="run gunicorn server")
    parsed_args = parser.parse_args()

    app = get_app()
    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
