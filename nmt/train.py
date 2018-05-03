# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""For training NMT models."""
from __future__ import print_function

import math
import os
import random
import time

import tensorflow as tf

# from . import attention_model
# from . import gnmt_model
# from . import inference
# from . import model as nmt_model
# from . import model_helper
# from .utils import misc_utils as utils
# from .utils import nmt_utils
import attention_model
import gnmt_model
import inference
import model as nmt_model
import model_helper
import misc_utils as utils
import nmt_utils

import pickle
import numpy as np
np.set_printoptions(threshold=np.nan)
from exp3S import Exp3S

utils.check_tensorflow_version()

__all__ = [
    "run_sample_decode",
    "run_internal_eval", "run_external_eval", "run_full_eval", "train"
]


def run_sample_decode(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, src_data, tgt_data):
  """Sample decode a random sentence from src_data."""
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer")

  _sample_decode(loaded_infer_model, global_step, infer_sess, hparams,
                 infer_model.iterator, src_data, tgt_data,
                 infer_model.src_placeholder,
                 infer_model.batch_size_placeholder, summary_writer)


def run_internal_eval(
    eval_model, eval_sess, model_dir, hparams, summary_writer):
  """Compute internal evaluation (perplexity) for both dev / test."""
  with eval_model.graph.as_default():
    loaded_eval_model, global_step = model_helper.create_or_load_model(
        eval_model.model, model_dir, eval_sess, "eval")

  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  dev_eval_iterator_feed_dict = {
      eval_model.src_file_placeholder: dev_src_file,
      eval_model.tgt_file_placeholder: dev_tgt_file
  }

  dev_ppl = _internal_eval(loaded_eval_model, global_step, eval_sess,
                           eval_model.iterator, dev_eval_iterator_feed_dict,
                           summary_writer, "dev")
  test_ppl = None
  if hparams.test_prefix:
    test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
    test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    test_eval_iterator_feed_dict = {
        eval_model.src_file_placeholder: test_src_file,
        eval_model.tgt_file_placeholder: test_tgt_file
    }
    test_ppl = _internal_eval(loaded_eval_model, global_step, eval_sess,
                              eval_model.iterator, test_eval_iterator_feed_dict,
                              summary_writer, "test")
  return dev_ppl, test_ppl


def run_external_eval(infer_model, infer_sess, model_dir, hparams,
                      summary_writer, save_best_dev=True):

  """Compute external evaluation (bleu, rouge, etc.) for both dev / test."""
  with infer_model.graph.as_default():
    loaded_infer_model, global_step = model_helper.create_or_load_model(
        infer_model.model, model_dir, infer_sess, "infer")

  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  dev_infer_iterator_feed_dict = {
      infer_model.src_placeholder: inference.load_data(dev_src_file),
      infer_model.batch_size_placeholder: hparams.infer_batch_size,
  }
  dev_scores = _external_eval(
      loaded_infer_model,
      global_step,
      infer_sess,
      hparams,
      infer_model.iterator,
      dev_infer_iterator_feed_dict,
      dev_tgt_file,
      "dev",
      summary_writer,
      save_on_best=save_best_dev)

  test_scores = None
  if hparams.test_prefix:
    test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
    test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    test_infer_iterator_feed_dict = {
        infer_model.src_placeholder: inference.load_data(test_src_file),
        infer_model.batch_size_placeholder: hparams.infer_batch_size,
    }
    test_scores = _external_eval(
        loaded_infer_model,
        global_step,
        infer_sess,
        hparams,
        infer_model.iterator,
        test_infer_iterator_feed_dict,
        test_tgt_file,
        "test",
        summary_writer,
        save_on_best=False)
  return dev_scores, test_scores, global_step


def run_full_eval(model_dir, infer_model, infer_sess, eval_model, eval_sess,
                  hparams, summary_writer, sample_src_data, sample_tgt_data):
  """Wrapper for running sample_decode, internal_eval and external_eval."""
  run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer,
                    sample_src_data, sample_tgt_data)
  dev_ppl, test_ppl = run_internal_eval(
      eval_model, eval_sess, model_dir, hparams, summary_writer)
  dev_scores, test_scores, global_step = run_external_eval(
      infer_model, infer_sess, model_dir, hparams, summary_writer)

  result_summary = _format_results("dev", dev_ppl, dev_scores, hparams.metrics)
  if hparams.test_prefix:
    result_summary += ", " + _format_results("test", test_ppl, test_scores,
                                             hparams.metrics)

  return result_summary, global_step, dev_scores, test_scores, dev_ppl, test_ppl


def train(hparams, scope=None, target_session=""):
  """Train a translation model."""
  log_device_placement = hparams.log_device_placement
  out_dir = hparams.out_dir
  num_train_steps = hparams.num_train_steps
  steps_per_stats = hparams.steps_per_stats
  steps_per_external_eval = hparams.steps_per_external_eval
  steps_per_eval = 10 * steps_per_stats
  if not steps_per_external_eval:
    steps_per_external_eval = 5 * steps_per_eval

  if not hparams.attention:
    model_creator = nmt_model.Model
  elif hparams.attention_architecture == "standard":
    model_creator = attention_model.AttentionModel
  elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
    model_creator = gnmt_model.GNMTModel
  else:
    raise ValueError("Unknown model architecture")

  train_model = model_helper.create_train_model(model_creator, hparams, scope)
  eval_model = model_helper.create_eval_model(model_creator, hparams, scope)
  infer_model = model_helper.create_infer_model(model_creator, hparams, scope)

  # Preload data for sample decoding.
  dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  sample_src_data = inference.load_data(dev_src_file)
  sample_tgt_data = inference.load_data(dev_tgt_file)

  summary_name = "train_log"
  model_dir = hparams.out_dir

  # Log and output files
  log_file = os.path.join(out_dir, "log_%d" % time.time())
  log_f = tf.gfile.GFile(log_file, mode="w")
  utils.print_out("# log_file=%s" % log_file, log_f)

  avg_step_time = 0.0

  # TensorFlow model
  config_proto = utils.get_config_proto(
      log_device_placement=log_device_placement)

  train_sess = tf.Session(
      target=target_session, config=config_proto, graph=train_model.graph)
  eval_sess = tf.Session(
      target=target_session, config=config_proto, graph=eval_model.graph)
  infer_sess = tf.Session(
      target=target_session, config=config_proto, graph=infer_model.graph)

  with train_model.graph.as_default():
    loaded_train_model, global_step = model_helper.create_or_load_model(
        train_model.model, model_dir, train_sess, "train")

  # Summary writer
  summary_writer = tf.summary.FileWriter(
      os.path.join(out_dir, summary_name), train_model.graph)

  # First evaluation
  run_full_eval(
      model_dir, infer_model, infer_sess,
      eval_model, eval_sess, hparams,
      summary_writer, sample_src_data,
      sample_tgt_data)

  last_stats_step = global_step
  last_eval_step = global_step
  last_external_eval_step = global_step

  # This is the training loop.
  step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
  checkpoint_total_count = 0.0
  speed, train_ppl = 0.0, 0.0
  start_train_time = time.time()

  utils.print_out(
      "# Start step %d, lr %g, %s" %
      (global_step, loaded_train_model.learning_rate.eval(session=train_sess),
       time.ctime()),
      log_f)

  # Initialize all of the iterators
  skip_count = hparams.batch_size * hparams.epoch_step
  utils.print_out("# Init train iterator, skipping %d elements" % skip_count)
  
  if hparams.curriculum == 'none':
    train_sess.run(
        train_model.iterator.initializer,
        feed_dict={
          train_model.skip_count_placeholder: skip_count
        })
  else:
    if hparams.curriculum == 'predictive_gain':
      exp3s = Exp3S(hparams.num_curriculum_buckets, 0.001, 0, 0.05)
    elif hparams.curriculum == 'look_back_and_forward':
      curriculum_point = 0

    handle = train_model.iterator.handle
    for i in range(hparams.num_curriculum_buckets):
      train_sess.run(
        train_model.iterator.initializer[i].initializer,
        feed_dict={
          train_model.skip_count_placeholder: skip_count
        })

    iterator_handles = [train_sess.run(train_model.iterator.initializer[i].string_handle(),
        feed_dict={
            train_model.skip_count_placeholder: skip_count
          })
      for i in range(hparams.num_curriculum_buckets)]

  utils.print_out("Starting training")

  while global_step < num_train_steps:
    ### Run a step ###
    start_time = time.time()
    try:
      if hparams.curriculum != 'none':
        if hparams.curriculum == 'predictive_gain':
          lesson = exp3s.draw_task()
        elif hparams.curriculum == 'look_back_and_forward':
          if curriculum_point == hparams.num_curriculum_buckets:
            lesson = np.random.randint(low=0, high=hparams.num_curriculum_buckets)
          else:
            lesson = curriculum_point if np.random.random_sample() < 0.8 else np.random.randint(low=0, high=hparams.num_curriculum_buckets)

        step_result = loaded_train_model.train(hparams, train_sess,
          handle=handle, iterator_handle=iterator_handles[lesson],
          use_fed_source_placeholder=loaded_train_model.use_fed_source,
          fed_source_placeholder=loaded_train_model.fed_source)

        (_, step_loss, step_predict_count, step_summary, global_step,
          step_word_count, batch_size, source) = step_result

        if hparams.curriculum == 'predictive_gain':
          new_loss = train_sess.run([loaded_train_model.train_loss],
            feed_dict={
              handle: iterator_handles[lesson],
              loaded_train_model.use_fed_source: True,
              loaded_train_model.fed_source: source
            })

          # new_loss = loaded_train_model.train_loss.eval(
          #   session=train_sess,
          #   feed_dict={
          #     handle: iterator_handles[lesson],
          #     loaded_train_model.use_fed_source: True,
          #     loaded_train_model.fed_source: source
          #   })

          # utils.print_out("lesson: %s, step loss: %s, new_loss: %s" % (lesson, step_loss, new_loss))
          # utils.print_out("exp3s dist: %s" % (exp3s.pi, ))

          curriculum_point_a = lesson * (hparams.src_max_len // hparams.num_curriculum_buckets) + 1
          curriculum_point_b = (lesson + 1) * (hparams.src_max_len // hparams.num_curriculum_buckets) + 1

          v = step_loss - new_loss
          exp3s.update_w(v, float(curriculum_point_a + curriculum_point_b)/2.0)
        elif hparams.curriculum == 'look_back_and_forward':
          utils.print_out("step loss: %s, lesson: %s" % (step_loss, lesson))
          curriculum_point_a = curriculum_point * (hparams.src_max_len // hparams.num_curriculum_buckets) + 1
          curriculum_point_b = (curriculum_point + 1) * (hparams.src_max_len // hparams.num_curriculum_buckets) + 1

          if step_loss < (hparams.curriculum_progress_loss * (float(curriculum_point_a + curriculum_point_b)/2.0)):
            curriculum_point += 1
      else:
        step_result = loaded_train_model.train(hparams, train_sess)
        (_, step_loss, step_predict_count, step_summary, global_step,
          step_word_count, batch_size) = step_result
      hparams.epoch_step += 1
    except tf.errors.OutOfRangeError:
      # Finished going through the training dataset.  Go to next epoch.
      hparams.epoch_step = 0
      # utils.print_out(
      #     "# Finished an epoch, step %d. Perform external evaluation" %
      #     global_step)
      # run_sample_decode(infer_model, infer_sess,
      #                   model_dir, hparams, summary_writer, sample_src_data,
      #                   sample_tgt_data)
      # dev_scores, test_scores, _ = run_external_eval(
      #     infer_model, infer_sess, model_dir,
      #     hparams, summary_writer)
      if hparams.curriculum == 'none':
        train_sess.run(
            train_model.iterator.initializer,
            feed_dict={
              train_model.skip_count_placeholder: 0
            })
      else:
        train_sess.run(
            train_model.iterator.initializer[lesson].initializer,
            feed_dict={
              train_model.skip_count_placeholder: 0
            })
      continue

    # Write step summary.
    summary_writer.add_summary(step_summary, global_step)

    # update statistics
    step_time += (time.time() - start_time)

    checkpoint_loss += (step_loss * batch_size)
    checkpoint_predict_count += step_predict_count
    checkpoint_total_count += float(step_word_count)

    # Once in a while, we print statistics.
    if global_step - last_stats_step >= steps_per_stats:
      if hparams.curriculum == 'predictive_gain':
        utils.print_out("lesson: %s, step loss: %s, new_loss: %s" % (lesson, step_loss, new_loss))
        utils.print_out("exp3s dist: %s" % (exp3s.pi, ))

      last_stats_step = global_step

      # Print statistics for the previous epoch.
      avg_step_time = step_time / steps_per_stats
      train_ppl = utils.safe_exp(checkpoint_loss / checkpoint_predict_count)
      speed = checkpoint_total_count / (1000 * step_time)
      utils.print_out(
          "  global step %d lr %g "
          "step-time %.2fs wps %.2fK ppl %.2f %s" %
          (global_step,
           loaded_train_model.learning_rate.eval(session=train_sess),
           avg_step_time, speed, train_ppl, _get_best_results(hparams)),
          log_f)
      
      if math.isnan(train_ppl):
        break

      # Reset timer and loss.
      step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
      checkpoint_total_count = 0.0

    if global_step - last_eval_step >= steps_per_eval:
      last_eval_step = global_step

      utils.print_out("# Save eval, global step %d" % global_step)
      utils.add_summary(summary_writer, global_step, "train_ppl", train_ppl)

      # Save checkpoint
      loaded_train_model.saver.save(
          train_sess,
          os.path.join(out_dir, "translate.ckpt"),
          global_step=global_step)

      # Evaluate on dev/test
      run_sample_decode(infer_model, infer_sess,
                        model_dir, hparams, summary_writer, sample_src_data,
                        sample_tgt_data)
      dev_ppl, test_ppl = run_internal_eval(
          eval_model, eval_sess, model_dir, hparams, summary_writer)

      dev_scores, test_scores, _ = run_external_eval(
          infer_model, infer_sess, model_dir,
          hparams, summary_writer)

    # if global_step - last_external_eval_step >= steps_per_external_eval:
    #   last_external_eval_step = global_step

    #   # Save checkpoint
    #   loaded_train_model.saver.save(
    #       train_sess,
    #       os.path.join(out_dir, "translate.ckpt"),
    #       global_step=global_step)
    #   run_sample_decode(infer_model, infer_sess,
    #                     model_dir, hparams, summary_writer, sample_src_data,
    #                     sample_tgt_data)
    #   dev_scores, test_scores, _ = run_external_eval(
    #       infer_model, infer_sess, model_dir,
    #       hparams, summary_writer)

  # Done training
  loaded_train_model.saver.save(
      train_sess,
      os.path.join(out_dir, "translate.ckpt"),
      global_step=global_step)

  result_summary, _, dev_scores, test_scores, dev_ppl, test_ppl = run_full_eval(
      model_dir, infer_model, infer_sess,
      eval_model, eval_sess, hparams,
      summary_writer, sample_src_data,
      sample_tgt_data)
  
  utils.print_out(
      "# Final, step %d lr %g "
      "step-time %.2f wps %.2fK ppl %.2f, %s, %s" %
      (global_step, loaded_train_model.learning_rate.eval(session=train_sess),
       avg_step_time, speed, train_ppl, result_summary, time.ctime()),
      log_f)
  utils.print_time("# Done training!", start_train_time)

  utils.print_out("# Start evaluating saved best models.")
  for metric in hparams.metrics:
    best_model_dir = getattr(hparams, "best_" + metric + "_dir")
    result_summary, best_global_step, _, _, _, _ = run_full_eval(
        best_model_dir, infer_model, infer_sess, eval_model, eval_sess, hparams,
        summary_writer, sample_src_data, sample_tgt_data)
    utils.print_out("# Best %s, step %d "
                    "step-time %.2f wps %.2fK, %s, %s" %
                    (metric, best_global_step, avg_step_time, speed,
                     result_summary, time.ctime()), log_f)

  summary_writer.close()
  return (dev_scores, test_scores, dev_ppl, test_ppl, global_step)


def _format_results(name, ppl, scores, metrics):
  """Format results."""
  result_str = "%s ppl %.2f" % (name, ppl)
  if scores:
    for metric in metrics:
      result_str += ", %s %s %.1f" % (name, metric, scores[metric])
  return result_str


def _get_best_results(hparams):
  """Summary of the current best results."""
  tokens = []
  for metric in hparams.metrics:
    tokens.append("%s %.2f" % (metric, getattr(hparams, "best_" + metric)))
  return ", ".join(tokens)


def _internal_eval(model, global_step, sess, iterator, iterator_feed_dict,
                   summary_writer, label):
  """Computing perplexity."""
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
  ppl = model_helper.compute_perplexity(model, sess, label)
  utils.add_summary(summary_writer, global_step, "%s_ppl" % label, ppl)
  return ppl


def _sample_decode(model, global_step, sess, hparams, iterator, src_data,
                   tgt_data, iterator_src_placeholder,
                   iterator_batch_size_placeholder, summary_writer):
  """Pick a sentence and decode."""
  iterator_feed_dict = {
      iterator_src_placeholder: src_data[-hparams.infer_batch_size:],
      iterator_batch_size_placeholder: hparams.infer_batch_size,
  }
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

  nmt_outputs, att_w_history, ext_w_history = model.decode(sess)

  if hparams.beam_width > 0:
    # get the top translation.
    nmt_outputs = nmt_outputs[0]

  nmt_outputs = np.asarray(nmt_outputs)

  outputs = []
  for i in range(hparams.infer_batch_size):
    tmp = {}
    translation = nmt_utils.get_translation(
        nmt_outputs,
        sent_id=i,
        tgt_sos=hparams.sos,
        tgt_eos=hparams.eos,
        bpe_delimiter=hparams.bpe_delimiter)
    if i <= 5:
      utils.print_out("    src: %s" % src_data[-hparams.infer_batch_size+i])
      utils.print_out("    ref: %s" % tgt_data[-hparams.infer_batch_size+i])
      utils.print_out(b"    nmt: %s" % translation)
    tmp['src'] = src_data[-hparams.infer_batch_size+i]
    tmp['ref'] = tgt_data[-hparams.infer_batch_size+i]
    tmp['nmt'] = translation
    if att_w_history is not None:
      tmp['attention_head'] = att_w_history[-hparams.infer_batch_size+i]
    if ext_w_history is not None:
      for j, ext_head in enumerate(ext_w_history):
        tmp['ext_head_{0}'.format(j)] = ext_head[-hparams.infer_batch_size+i]
    outputs.append(tmp)

  if hparams.record_w_history:
    with open(hparams.out_dir + '/heads_step_{0}.pickle'.format(global_step), 'wb') as f:
      if len(outputs) > 0:
        pickle.dump(outputs, f)

  # utils.print_out(pickle.dumps(outputs), f=hparams.out_dir + '/heads_step_{0}.pickle'.format(i))

def _external_eval(model, global_step, sess, hparams, iterator,
                   iterator_feed_dict, tgt_file, label, summary_writer,
                   save_on_best):
  """External evaluation such as BLEU and ROUGE scores."""
  out_dir = hparams.out_dir
  decode = global_step > 0
  if decode:
    utils.print_out("# External evaluation, global step %d" % global_step)

  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

  output = os.path.join(out_dir, "output_%s" % label)
  scores = nmt_utils.decode_and_evaluate(
      label,
      model,
      sess,
      output,
      ref_file=tgt_file,
      metrics=hparams.metrics,
      bpe_delimiter=hparams.bpe_delimiter,
      beam_width=hparams.beam_width,
      tgt_sos=hparams.sos,
      tgt_eos=hparams.eos,
      decode=decode)
  # Save on best metrics
  if decode:
    for metric in hparams.metrics:
      utils.add_summary(summary_writer, global_step, "%s_%s" % (label, metric),
                        scores[metric])
      # metric: larger is better
      if save_on_best and scores[metric] > getattr(hparams, "best_" + metric):
        setattr(hparams, "best_" + metric, scores[metric])
        model.saver.save(
            sess,
            os.path.join(
                getattr(hparams, "best_" + metric + "_dir"), "translate.ckpt"),
            global_step=model.global_step)
    utils.save_hparams(out_dir, hparams)
  return scores
