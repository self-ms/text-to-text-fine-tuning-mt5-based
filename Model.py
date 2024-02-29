from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import logging
import torch
from ArgumentsHandler import ModelArguments, DataTrainingArguments
from DatasetHandler import Dtataframe2Dataset, T2TDataCollator
import os

class MT5 :
    logger = logging.getLogger(__name__)
    def __init__(self, args):

        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(args))
        self.logger.info("*** Converting csv to nlp dataset ***")
        Dtataframe2Dataset(data_args)

        if (os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir):
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        )
        self.logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,)

        self.logger.info("Training/evaluation parameters %s", training_args)

        set_seed(training_args.seed)

        tokenizer = MT5Tokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        tokenizer.save_pretrained(training_args.output_dir)

        model = MT5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        # Get datasets
        self.logger.info('loading data')
        train_dataset  = torch.load(data_args.train_file_path)
        valid_dataset = torch.load(data_args.valid_file_path)


        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=T2TDataCollator(),
        )

        # Training
        if training_args.do_train:
            trainer.save_model()
            trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)
            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(training_args.output_dir)

        # Evaluation
        results = {}
        if training_args.do_eval and training_args.local_rank in [-1, 0]:
            self.logger.info("*** Evaluate ***")

            eval_output = trainer.evaluate()

            output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                self.logger.info("***** Eval results *****")
                for key in sorted(eval_output.keys()):
                    self.logger.info("  %s = %s", key, str(eval_output[key]))
                    writer.write("%s = %s\n" % (key, str(eval_output[key])))

            results.update(eval_output)
            self.logger.info(results)