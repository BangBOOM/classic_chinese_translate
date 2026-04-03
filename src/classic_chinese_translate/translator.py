"""Fairseq model loading and translation inference."""

from collections import namedtuple

import torch
from fairseq import options, tasks, utils

Batch = namedtuple("Batch", "ids src_tokens src_lengths")


def _make_batches(lines, args, task, max_positions):
    tokens = [
        task.source_dictionary.encode_line(src_str, add_if_not_exist=False).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch["id"],
            src_tokens=batch["net_input"]["src_tokens"],
            src_lengths=batch["net_input"]["src_lengths"],
        )


class FairseqTranslator:
    """Wraps a fairseq translation model for inference."""

    def __init__(self, data_path, model_path, beam=5):
        self.args = self._parse_args(data_path, model_path, beam)
        self._load()

    def _parse_args(self, data_path, model_path, beam):
        parser = options.get_generation_parser(interactive=True)
        arg_list = [
            str(data_path),
            "--path",
            str(model_path),
            "--beam",
            str(beam),
            "--remove-bpe",
        ]
        args = options.parse_args_and_arch(parser, input_args=arg_list)
        return args

    def _load(self):
        args = self.args
        utils.import_user_module(args)

        if args.buffer_size < 1:
            args.buffer_size = 1
        if args.max_tokens is None and args.max_sentences is None:
            args.max_sentences = 1

        print(args)

        self.use_cuda = torch.cuda.is_available() and not args.cpu

        # Setup task
        self.task = tasks.setup_task(args)

        # Load ensemble
        print("| loading model(s) from {}".format(args.path))
        self.models, _model_args = utils.load_ensemble_for_inference(
            args.path.split(":"),
            self.task,
            model_arg_overrides=eval(args.model_overrides),
        )

        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        # Optimize ensemble for generation
        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment,
            )
            if args.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()

        # Initialize generator
        self.generator = self.task.build_generator(args)

        # Load alignment dictionary
        self.align_dict = utils.load_align_dict(args.replace_unk)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in self.models],
        )

    def translate(self, tokenized_inputs):
        """Run inference on a list of BPE-tokenized sentences, return first result.

        Args:
            tokenized_inputs: list of strings like ['袁州 火 , 龙庆 , ...']

        Returns:
            Translated string with spaces removed.
        """
        args = self.args
        start_id = 0
        results = []
        for batch in _make_batches(tokenized_inputs, args, self.task, self.max_positions):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
            }
            translations = self.task.inference_step(self.generator, self.models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, args.remove_bpe)

            for hypo in hypos[: min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=(
                        hypo["alignment"].int().cpu()
                        if hypo["alignment"] is not None
                        else None
                    ),
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=args.remove_bpe,
                )
                return hypo_str
