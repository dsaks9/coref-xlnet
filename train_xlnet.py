#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time

import tensorflow as tf
import util_xlnet
import logging
from absl import flags

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Model
flags.DEFINE_string("model_config_path", default=None,
      help="Model config path.")
flags.DEFINE_float("dropout", default=0.1,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
      help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")
flags.DEFINE_string("summary_type", default="last",
      help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_bool("use_summ_proj", default=True,
      help="Whether to use projection for summarizing sequences.")
flags.DEFINE_bool("use_bfloat16", False,
      help="Whether to use bfloat16.")
flags.DEFINE_integer("mem_len", default=0,
      help="Number of steps to cache")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")

# I/O paths
flags.DEFINE_bool("overwrite_data", default=False,
      help="If False, will use cached data if available.")
flags.DEFINE_string("init_checkpoint", default=None,
      help="checkpoint path for initializing the model. "
      "Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("output_dir", default="",
      help="Output dir for TF records.")
flags.DEFINE_string("spiece_model_file", default="",
      help="Sentence Piece model path.")
flags.DEFINE_string("model_dir", default="",
      help="Directory for saving the finetuned model.")
flags.DEFINE_string("data_dir", default="",
      help="Directory for input data.")

# TPUs and machines
flags.DEFINE_bool("use_tpu", default=False, help="whether to use TPU.")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", default=8,
      help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context "
      "of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default=None, help="TPU name.")
flags.DEFINE_string("tpu_zone", default=None, help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("iterations", default=1000,
      help="number of iterations per TPU training loop.")

# training
flags.DEFINE_bool("do_train", default=True, help="whether to do training")
flags.DEFINE_integer("train_steps", default=1000,
      help="Number of training steps")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_float("learning_rate", default=1e-5, help="initial learning rate")
flags.DEFINE_float("lr_layer_decay_rate", 1.0,
                   "Top layer: lr[L] = FLAGS.learning_rate."
                   "Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float("min_lr_ratio", default=0.0,
      help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_integer("max_save", default=0,
      help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("save_steps", default=None,
      help="Save the model for every save_steps. "
      "If None, not to save any model.")
flags.DEFINE_integer("train_batch_size", default=8,
      help="Batch size for training")
flags.DEFINE_float("weight_decay", default=0.00, help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-8, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")

# evaluation
flags.DEFINE_bool("do_eval", default=False, help="whether to do eval")
flags.DEFINE_bool("do_predict", default=False, help="whether to do prediction")
flags.DEFINE_float("predict_threshold", default=0,
      help="Threshold for binary prediction.")
flags.DEFINE_string("eval_split", default="dev", help="could be dev or test")
flags.DEFINE_integer("eval_batch_size", default=128,
      help="batch size for evaluation")
flags.DEFINE_integer("predict_batch_size", default=128,
      help="batch size for prediction.")
flags.DEFINE_string("predict_dir", default=None,
      help="Dir for saving prediction files.")
flags.DEFINE_bool("eval_all_ckpt", default=False,
      help="Eval all ckpts. If False, only evaluate the last one.")
flags.DEFINE_string("predict_ckpt", default=None,
      help="Ckpt path for do_predict. If None, use the last one.")

# task specific
flags.DEFINE_string("task_name", default=None, help="Task name")
flags.DEFINE_integer("max_seq_length", default=128, help="Max sequence length")
flags.DEFINE_integer("shuffle_buffer", default=2048,
      help="Buffer size used for shuffle.")
flags.DEFINE_integer("num_passes", default=1,
      help="Num passes for processing training data. "
      "This is use to batch data without loss for TPUs.")
flags.DEFINE_bool("uncased", default=False,
      help="Use uncased.")
flags.DEFINE_string("cls_scope", default=None,
      help="Classifier layer scope.")
flags.DEFINE_bool("is_regression", default=False,
      help="Whether it's a regression task.")

FLAGS = flags.FLAGS

if __name__ == "__main__":
  FLAGS(sys.argv)
  config = util_xlnet.initialize_from_env()

  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]

  model = util_xlnet.get_model(config, FLAGS)
  print("####################################")
  print("####################################")
  print(model)
  print("####################################")

  saver = tf.train.Saver()

  log_dir = config["log_dir"]
  max_steps = config['num_epochs'] * config['num_docs']
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)

  max_f1 = 0
  mode = 'w'

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    model.start_enqueue_thread(session)
    accumulated_loss = 0.0

    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print("Restoring from: {}".format(ckpt.model_checkpoint_path))
      saver.restore(session, ckpt.model_checkpoint_path)
      mode = 'a'
    fh = logging.FileHandler(os.path.join(log_dir, 'stdout.log'), mode=mode)
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)

    initial_time = time.time()
    while True:
      tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
      accumulated_loss += tf_loss

      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        logger.info("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
        writer.add_summary(util_xlnet.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0

      if tf_global_step  > 0 and tf_global_step % eval_frequency == 0:
        saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        eval_summary, eval_f1 = model.evaluate(session, tf_global_step)

        if eval_f1 > max_f1:
          max_f1 = eval_f1
          util_xlnet.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

        writer.add_summary(eval_summary, tf_global_step)
        writer.add_summary(util_xlnet.make_summary({"max_eval_f1": max_f1}), tf_global_step)

        logger.info("[{}] evaL_f1={:.4f}, max_f1={:.4f}".format(tf_global_step, eval_f1, max_f1))
        if tf_global_step > max_steps:
          break
