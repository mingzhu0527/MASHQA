from run_sentqa_f1_65 import *
from tensorflow.keras.layers import Dense
from os.path import join

flags.DEFINE_bool("do_eval", default=False, help="whether to do eval")
flags.DEFINE_bool("eval_all_ckpt", default=False,
      help="Eval all ckpts. If False, only evaluate the last one.")
flags.DEFINE_string("eval_file", default="",
                    help="Path of eval file.")
flags.DEFINE_float("temperature", default=10.0, help="Gumbel temperature")
flags.DEFINE_float("min_temperature", default=1.0, help="Min gumbel temperature")
flags.DEFINE_float("temp_decay_rate", default=3e-3, help="Gumbel temperature decay rate")

# 1. sent_list, 跟"context"在同一个level
# 2. answer_span，跟answer_start在同一个level
# 3. 跟answer_start在同一个level is None for QA with non_consec 

# from run_squad_GPU import *

class AttentionNoPoolingLayer(object):
    
    def __init__(self):
        self.output_layer = None
        
    def __call__(self, k_enc_outputs, q_enc_outputs, v_enc_outputs):
        # k -- embedding dimension
        
        k = k_enc_outputs.shape[2].value
        # dot-product attention
        with tf.variable_scope('attention_layer'):
            # K * Q_t
            proj_q = tf.matmul(k_enc_outputs, q_enc_outputs, transpose_b=True)
            # Normalize (scaled dot product)
            proj_q = tf.nn.softmax(tf.multiply(proj_q, 1/math.sqrt(4.0*k)))
            # (K * Q_t) * Q
            proj_q = tf.matmul(proj_q, v_enc_outputs)
        
        return proj_q


    
class AttentionPoolingParameterizedLayer(object):
   
    def __init__(self):
        self.output_layer = None
       
    def __call__(self, k_enc_outputs, q_enc_outputs):
        # k -- embedding dimension
        k = k_enc_outputs.shape[2].value
        q = q_enc_outputs.shape[2].value
        # dot-product attention
        with tf.variable_scope('attention_layer'):
            # K_ = K * W_k
            self.W_k = Dense(k)
            k_enc_outputs = self.W_k(k_enc_outputs)
            # Q_ = Q * W_q
            self.W_q = Dense(q)
            q_enc_outputs = self.W_q(q_enc_outputs)
            # K * Q_t
            proj_q0 = tf.matmul(k_enc_outputs, q_enc_outputs, transpose_b=True)
            # Normalize
#             proj_q = tf.nn.softmax(tf.multiply(proj_q, 1/math.sqrt(k)))
            proj_q1 = tf.nn.softmax(tf.multiply(proj_q0, 1.0/(4.0*k)))
            # (K * Q_t) * Q
            proj_q = tf.matmul(proj_q1, q_enc_outputs)
            proj_q = tf.transpose(proj_q, perm=[0, 2, 1])
        # pooling
        with tf.variable_scope('pooling_layer'):
            self.output_layer = Dense(1)
            proj_q = self.output_layer(proj_q)
            proj_q = tf.squeeze(proj_q, axis=2)
       
        return proj_q, tf.reshape(proj_q0, [-1, 1024]), tf.reshape(proj_q1, [-1, 1024])

    
    
# class GumbelAttentionNoPoolingParameterizedLayer(object):
   
#     def __init__(self):
#         self.output_layer = None
       
#     def __call__(self, k_enc_outputs, q_enc_outputs):
#         # k -- embedding dimension
#         k = k_enc_outputs.shape[2].value
#         q = q_enc_outputs.shape[2].value
#         # dot-product attention
#         with tf.variable_scope('attention_layer'):
#             # K_ = K * W_k
#             self.W_k = Dense(k)
#             k_enc_outputs = self.W_k(k_enc_outputs)
#             # Q_ = Q * W_q
#             self.W_q = Dense(q)
#             q_enc_outputs = self.W_q(q_enc_outputs)
#             # K * Q_t
#             proj_q0 = tf.matmul(k_enc_outputs, q_enc_outputs, transpose_b=True)
#             # Normalize
# #             proj_q = tf.nn.softmax(tf.multiply(proj_q, 1/math.sqrt(k)))
#             proj_q1 = tf.nn.softmax(tf.multiply(proj_q0, 1.0/(4.0*k)))
#             # (K * Q_t) * Q
#             proj_q = tf.matmul(proj_q1, q_enc_outputs)
       
#         return proj_q, tf.reshape(proj_q0, [-1, 1024]), tf.reshape(proj_q1, [-1, 1024])
    
class GumbelAttentionNoPoolingParameterizedLayer(object):
   
    def __init__(self):
        self.output_layer = None
       
    def __call__(self, k_enc_outputs, q_enc_outputs, temperature):
        # k -- embedding dimension
        # b l h
        k = k_enc_outputs.shape[2].value
        q = q_enc_outputs.shape[2].value
        # dot-product attention
        with tf.variable_scope('gumbel_attention_layer'):
            # K_ = K * W_k
            self.W_k = Dense(k)
            k_enc_outputs = self.W_k(k_enc_outputs)
            # Q_ = Q * W_q
            self.W_q = Dense(q)
            q_enc_outputs = self.W_q(q_enc_outputs)
            # K * Q_t
            # b l l
            proj_q0 = tf.matmul(k_enc_outputs, q_enc_outputs, transpose_b=True)
            proj_q1, proj_q2 = gumbel_softmax(tf.multiply(proj_q0, 1/(k)), temperature, hard=False)
            # b l h
            proj_q = tf.matmul(proj_q1, q_enc_outputs)
#             proj_q = tf.transpose(proj_q, perm=[0, 2, 1])
        return proj_q, proj_q0, proj_q1, proj_q2


# Gumbel softmax function, taken from Eric Jang's implementation: https://github.com/ericjang/gumbel-softmax
def sample_gumbel(shape, eps=1e-20):
    """ Sample from Gumbel(0, 1) """
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution """
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y/temperature), y

def gumbel_softmax(logits, temperature, hard=False):
    """ Sample from Gumbel-Softmax distribution and optionally discretize
    if hard=True, take argmax, but differentiate w.r.t. soft sample y. 
    The returned sample will be one-hot if hard=true, 
    otherwise probability distribution that sums to 1
    """
    y, x = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y, x

def get_qa_outputs(FLAGS, features, is_training):
    """Loss for downstream span-extraction QA tasks such as SQuAD."""

    inp = tf.transpose(features["input_ids"], [1, 0])
    seg_id = tf.transpose(features["segment_ids"], [1, 0])
    inp_mask = tf.transpose(features["input_mask"], [1, 0])
    cls_index = tf.reshape(features["cls_index"], [-1])
#     print("features['sent_labels']", features['sent_labels'])
    
    seq_len = tf.shape(inp)[0]

    xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    run_config = xlnet.create_run_config(is_training, True, FLAGS)

    xlnet_model = xlnet.XLNetModel(
            xlnet_config=xlnet_config,
            run_config=run_config,
            input_ids=inp,
            seg_ids=seg_id,
            input_mask=inp_mask)
    output = xlnet_model.get_sequence_output()
    initializer = xlnet_model.get_initializer()

    return_dict = {}

    # invalid position mask such as query and special symbols (PAD, SEP, CLS)
    p_mask = features["p_mask"]

    keep_prob = 1.0
    if is_training:
        keep_prob = 0.8
    else:
        keep_prob = 1.0
        
    bsz = tf.shape(output)[1]
    hsz = tf.shape(output)[2]
    # output shape: lbh, l = 512, b = batchsize, h = hiddensize = 1024
    # reshape into 16*batchsize*32h, then condense 32h into sentence vector
    
    sent_output = tf.reshape(output, [-1, FLAGS.max_sent_length, bsz, hsz])
    nsent = tf.shape(sent_output)[0]
    sent_output = tf.einsum("nsbh->nbsh", sent_output)
    sent_output = tf.reshape(sent_output, [nsent, bsz, 32*1024])
    sent_rep_vectors = tf.layers.dense(
                sent_output,
                1024,
                kernel_initializer=initializer,
                name="sent_dense_1")
    sent_rep_vectors = tf.nn.relu(sent_rep_vectors)
    sent_rep_vectors = tf.nn.dropout(sent_rep_vectors, keep_prob)
    cls_index = tf.one_hot(cls_index, seq_len, axis=-1, dtype=tf.float32)
    cls_feature = tf.einsum("lbh,bl->bh", output, cls_index)
    # 1, b, h
    cls_feature = tf.expand_dims(cls_feature, 0)
    cls_tile = tf.tile(cls_feature, [nsent, 1, 1])
    # n, b, 2h
    sent_rep_cls = tf.concat([sent_rep_vectors, cls_tile], -1)
    
    # 13, b, 2h
    sent_cls_slice = tf.slice(sent_rep_cls, [0, 0, 0], [FLAGS.max_sent_num, bsz, 2048])
    
#     global_step = tf.train.get_or_create_global_step()
#     decay_temp = tf.train.polynomial_decay(
#                 FLAGS.temperature,
#                 global_step=global_step,
#                 decay_steps=FLAGS.train_steps,
#                 end_learning_rate=FLAGS.min_temperature,
#                 power=1.0)
#     # b, 13, 2h
#     sent_cls_slice = tf.transpose(sent_cls_slice, [1, 0, 2])
#     sent_cls_slice_out = sent_cls_slice
#     # b, 13, 2h
#     sent_cls_slice_gumbel, proj_q0, proj_q1, proj_q2 = GumbelAttentionNoPoolingParameterizedLayer()(sent_cls_slice, sent_cls_slice, decay_temp)
#     # b, 13, 2h
#     sent_cls_slice = tf.transpose(sent_cls_slice_gumbel, [1, 0, 2])
    
    # logit of the start position
    with tf.variable_scope("class_logits"):
        
        
        start_logits = tf.layers.dense(
                sent_cls_slice,
                512,
                kernel_initializer=initializer,
                name="start_dense_1")
        start_logits = tf.nn.relu(start_logits)
        start_logits = tf.nn.dropout(start_logits, keep_prob)
        
        start_logits = tf.layers.dense(
                start_logits,
                384,
                kernel_initializer=initializer,
                name="start_dense_2")
        start_logits = tf.nn.relu(start_logits)
        start_logits = tf.nn.dropout(start_logits, keep_prob)
        
        start_logits = tf.layers.dense(
                start_logits,
                2,
                kernel_initializer=initializer,
                name="start_dense_3")
        
        start_logits = tf.transpose(start_logits, [1, 0, 2])
#         start_logits_masked = start_logits * (1 - p_mask) - 1e30 * p_mask
        start_log_probs = tf.nn.log_softmax(start_logits, -1)


    return_dict["start_log_probs"] = start_log_probs
    return_dict["proj_q0"] = proj_q0
    return_dict["proj_q1"] = proj_q1
    return_dict["proj_q2"] = proj_q2
    return_dict["sent_cls_slice_gumbel"] = sent_cls_slice_gumbel
#     if is_training:
#         return_dict["start_log_probs"] = start_log_probs
# #         return_dict["end_log_probs"] = end_log_probs
#     else:
#         return_dict["start_top_log_probs"] = start_top_log_probs
#         return_dict["start_top_index"] = start_top_index
#         return_dict["end_top_log_probs"] = end_top_log_probs
#         return_dict["end_top_index"] = end_top_index

    

    return return_dict


def input_fn_builder(input_glob, seq_length, is_training, drop_remainder,
                                         num_hosts, num_threads=8):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
            "unique_ids": tf.FixedLenFeature([], tf.int64),
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "cls_index": tf.FixedLenFeature([], tf.int64),
#             "span_sent_start_poses":tf.FixedLenFeature([], tf.int64),
#             "span_sent_end_poses":tf.FixedLenFeature([], tf.int64),
            "p_mask": tf.FixedLenFeature([seq_length], tf.float32)
    }

#     if is_training:
    name_to_features["sent_labels"] = tf.FixedLenFeature([FLAGS.max_sent_num], tf.int64)
#         name_to_features["start_poses"] = tf.FixedLenFeature([], tf.int64)
#         name_to_features["end_poses"] = tf.FixedLenFeature([], tf.int64)
#         name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
#         name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
#         name_to_features["is_impossible"] = tf.FixedLenFeature([], tf.float32)

    tf.logging.info("Input tfrecord file glob {}".format(input_glob))
    global_input_paths = tf.gfile.Glob(input_glob)
    tf.logging.info("Find {} input paths {}".format(
            len(global_input_paths), global_input_paths))

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        if FLAGS.use_tpu:
            batch_size = params["batch_size"]
        elif is_training:
            batch_size = FLAGS.train_batch_size
        else:
            batch_size = FLAGS.predict_batch_size

        # Split tfrecords across hosts
        if num_hosts > 1:
            host_id = params["context"].current_host
            num_files = len(global_input_paths)
            if num_files >= num_hosts:
                num_files_per_host = (num_files + num_hosts - 1) // num_hosts
                my_start_file_id = host_id * num_files_per_host
                my_end_file_id = min((host_id + 1) * num_files_per_host, num_files)
                input_paths = global_input_paths[my_start_file_id: my_end_file_id]
            tf.logging.info("Host {} handles {} files".format(host_id, len(input_paths)))
        else:
            input_paths = global_input_paths

        if len(input_paths) == 1:
            d = tf.data.TFRecordDataset(input_paths[0])
            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            if is_training:
                d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)
                d = d.repeat()
        else:
            d = tf.data.Dataset.from_tensor_slices(input_paths)
            # file level shuffle
            d = d.shuffle(len(input_paths)).repeat()

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_threads, len(input_paths))

            d = d.apply(
                    tf.contrib.data.parallel_interleave(
                            tf.data.TFRecordDataset,
                            sloppy=is_training,
                            cycle_length=cycle_length))

            if is_training:
                # sample level shuffle
                d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)

        d = d.apply(
                tf.contrib.data.map_and_batch(
                        lambda record: _decode_record(record, name_to_features),
                        batch_size=batch_size,
                        num_parallel_batches=num_threads,
                        drop_remainder=drop_remainder))
        d = d.prefetch(1024)

        return d

    return input_fn


def get_model_fn():
    def model_fn(features, labels, mode, params):
        #### Training or Evaluation
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        #### Get loss from inputs
        outputs = get_qa_outputs(FLAGS, features, is_training)

        #### Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        scaffold_fn = None

        #### Evaluation mode
        if mode == tf.estimator.ModeKeys.PREDICT:
            if FLAGS.init_checkpoint:
                tf.logging.info("init_checkpoint not being used in predict mode.")

            predictions = {
                    "unique_ids": features["unique_ids"],
                    "sent_labels": features["sent_labels"],
                    "start_top_log_probs": outputs["start_log_probs"],
                    "proj_q0": outputs["proj_q0"],
                    "proj_q1": outputs["proj_q1"],
                    "proj_q2": outputs["proj_q2"],
                    "sent_cls_slice_gumbel": outputs["sent_cls_slice_gumbel"],
#                     "end_top_index": outputs["end_top_index"],
#                     "end_top_log_probs": outputs["end_top_log_probs"],
#                     "cls_logits": outputs["cls_logits"]
            }

            
            if FLAGS.use_tpu:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.estimator.EstimatorSpec(
                        mode=mode, predictions=predictions)
            return output_spec

        ### Compute loss
        seq_length = tf.shape(features["input_ids"])[1]
        
        
        class_weights = tf.constant([[1.0, 20.0]])
        one_hot_target = tf.one_hot(features["sent_labels"], 2, dtype=tf.float32)
        weights = tf.reduce_sum(class_weights * one_hot_target, axis=-1)
        per_example_loss = -tf.reduce_sum(outputs["start_log_probs"] * one_hot_target, -1)
        weighted_losses = per_example_loss * weights
        total_loss = tf.reduce_mean(per_example_loss)
        
#         one_hot_target = tf.one_hot(features["sent_labels"], 2, dtype=tf.float32)
#         per_example_loss = -tf.reduce_sum(outputs["start_log_probs"] * one_hot_target, -1)
#         total_loss = tf.reduce_mean(per_example_loss)

        #### Configuring the optimizer
        train_op, learning_rate, _ = model_utils.get_train_op(FLAGS, total_loss)

        monitor_dict = {}
        monitor_dict["lr"] = learning_rate

        #### load pretrained models
        scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

        #### Constucting training TPUEstimatorSpec with new cache.
        if FLAGS.use_tpu:
            host_call = function_builder.construct_scalar_host_call(
                    monitor_dict=monitor_dict,
                    model_dir=FLAGS.model_dir,
                    prefix="train/",
                    reduce_fn=tf.reduce_mean)

            train_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
                    scaffold_fn=scaffold_fn)
        else:
            train_spec = tf.estimator.EstimatorSpec(
                    mode=mode, loss=total_loss, train_op=train_op)

        return train_spec

    return model_fn

def _get_spm_basename():
    spm_basename = os.path.basename(FLAGS.spiece_model_file)
    return spm_basename

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    if FLAGS.do_prepro:
        preprocess()
        return

    #### Validate flags
    if FLAGS.save_steps is not None:
        FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError(
                "At least one of `do_train` and `do_predict` must be True.")

    if FLAGS.do_predict and not tf.gfile.Exists(FLAGS.predict_dir):
        tf.gfile.MakeDirs(FLAGS.predict_dir)

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(FLAGS.spiece_model_file)

    ### TPU Configuration
    run_config = model_utils.configure_tpu(FLAGS)

    model_fn = get_model_fn()
    spm_basename = _get_spm_basename()

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    if FLAGS.use_tpu:
        estimator = tf.contrib.tpu.TPUEstimator(
                use_tpu=FLAGS.use_tpu,
                model_fn=model_fn,
                config=run_config,
                train_batch_size=FLAGS.train_batch_size,
                predict_batch_size=FLAGS.predict_batch_size)
    else:
        estimator = tf.estimator.Estimator(
                model_fn=model_fn,
                config=run_config)

    if FLAGS.do_train:
        train_rec_glob = os.path.join(
                FLAGS.output_dir,
                "{}.*.slen-{}.qlen-{}.train.tf_record".format(
                spm_basename, FLAGS.max_seq_length,
                FLAGS.max_query_length))

        train_input_fn = input_fn_builder(
                input_glob=train_rec_glob,
                seq_length=FLAGS.max_seq_length,
                is_training=True,
                drop_remainder=True,
                num_hosts=FLAGS.num_hosts)

        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

    if FLAGS.do_predict:
        eval_examples = read_squad_examples(FLAGS.predict_file, is_training=False)

        with tf.gfile.Open(FLAGS.predict_file) as f:
            orig_data = json.load(f)["data"]

        eval_rec_file = os.path.join(
                FLAGS.output_dir,
                "{}.slen-{}.qlen-{}.eval.tf_record".format(
                        spm_basename, FLAGS.max_seq_length, FLAGS.max_query_length))
        eval_feature_file = os.path.join(
                FLAGS.output_dir,
                "{}.slen-{}.qlen-{}.eval.features.pkl".format(
                        spm_basename, FLAGS.max_seq_length, FLAGS.max_query_length))

        if tf.gfile.Exists(eval_rec_file) and tf.gfile.Exists(
                eval_feature_file) and not FLAGS.overwrite_data:
            tf.logging.info("Loading eval features from {}".format(eval_feature_file))
            with tf.gfile.Open(eval_feature_file, 'rb') as fin:
                eval_features = pickle.load(fin)
        else:
            eval_writer = FeatureWriter(filename=eval_rec_file, is_training=False)
            eval_features = []

            def append_feature(feature):
                eval_features.append(feature)
                eval_writer.process_feature(feature)

            convert_examples_to_features(
                    examples=eval_examples,
                    sp_model=sp_model,
                    max_seq_length=FLAGS.max_seq_length,
                    doc_stride=FLAGS.doc_stride,
                    max_query_length=FLAGS.max_query_length,
                    is_training=False,
                    output_fn=append_feature)
            eval_writer.close()

            with tf.gfile.Open(eval_feature_file, 'wb') as fout:
                pickle.dump(eval_features, fout)

        eval_input_fn = input_fn_builder(
                input_glob=eval_rec_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=False,
                num_hosts=1)

        steps_and_files = []
        filenames = tf.gfile.ListDirectory(FLAGS.model_dir)

        for filename in filenames:
            if filename.endswith(".index"):
                ckpt_name = filename[:-6]
                cur_filename = join(FLAGS.model_dir, ckpt_name)
                global_step = int(cur_filename.split("-")[-1])
                tf.logging.info("Add {} to eval list.".format(cur_filename))
                steps_and_files.append([global_step, cur_filename])
        steps_and_files = sorted(steps_and_files, key=lambda x: x[0])
        if not FLAGS.eval_all_ckpt:
            for filename in filenames:
                if filename.endswith(".index"):
                    ckpt_name = filename[:-6]
                    cur_filename = join(FLAGS.model_dir, ckpt_name)
                    global_step = int(cur_filename.split("-")[-1])
                    if global_step == FLAGS.train_steps:
                        tf.logging.info("Add {} to eval list.".format(cur_filename))
                        steps_and_files.append([global_step, cur_filename])
                        break
            steps_and_files = steps_and_files[-1:]
        
        eval_results = []
        ep = FLAGS.output_dir.split('/')
        eval_path = ''
        if len(ep) == 2:
            eval_path = ep[1]
        md = FLAGS.model_dir.split('/')
        md_path = ''
        if len(md) == 2:
            md_path = md[1]
        result_fn = "results_" + md_path + "_" + eval_path + "_" + str(FLAGS.train_steps) + ".txt"
        with tf.gfile.Open(os.path.join(FLAGS.predict_dir, result_fn), "w") as res_out:
            res_out.write("step\t precision\t recall\t f1\n")
            for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
                output_fn = "predictions_" + md_path + "_" + eval_path + "_" + str(global_step) + ".tsv"
                with tf.gfile.Open(os.path.join(FLAGS.predict_dir, output_fn), "w") as fout:
#                     cur_results = {}
                    global_prec = 0
                    global_recc = 0
                    global_correct = 0
                    for pred_cnt, result in enumerate(estimator.predict(
                                input_fn=eval_input_fn,
                                checkpoint_path=filename,
                                yield_single_examples=False)):

                        if pred_cnt % 100 == 0:
                            tf.logging.info("Processing example: %d" % (pred_cnt))
                        unique_id = result["unique_ids"] # length equal to batch
                        sent_labs = result["sent_labels"].flatten() # 32*13

                        num_ans = np.sum(sent_labs, axis=-1)
                        prob_array = result["start_top_log_probs"] #32 * 13 * 2
        #                 print(prob_array.shape)
                        pred_ans = prob_array[:, :, 1] - prob_array[:, :, 0]
                        pred_ans[pred_ans<=0] = 0
                        pred_ans[pred_ans>0] = 1
                        pred_ans = pred_ans.flatten()

                        for i, (ans, lab) in enumerate(zip(pred_ans, sent_labs)):
                            fout.write("{}\t{}\t{}\t{}\n".format(pred_cnt, i, int(ans), lab))

                        temp_sum = float(sum(x[0] * x[1] for x in zip(sent_labs, pred_ans)))
                        global_correct += temp_sum
                        global_prec += sum(pred_ans)
                        global_recc += num_ans

                    # Log current result
                precision = 0
                recall = 0
                f1 = 0
                if global_prec != 0:
                    precision = global_correct/global_prec
                if global_recc !=0:
                    recall = global_correct/global_recc
                if precision + recall != 0:
                    f1 = 2*(precision * recall)/(precision + recall)
                ret_dict = {'precision':precision, 'recall':recall, 'f1':f1, 'global_step':global_step}
                eval_results.append(ret_dict)
                res_out.write("{}\t{}\t{}\t{}\n".format(global_step, precision, recall, f1))
                tf.logging.info("=" * 10)
                log_str = ''
                for key, val in sorted(eval_results[-1].items(), key=lambda x: x[0]): 
                    log_str += "{} {} | ".format(key, val)
                tf.logging.info("=" * 10)
                tf.logging.info(log_str)
            eval_results.sort(key=lambda x: x['f1'], reverse=True)

            tf.logging.info("=" * 20)
            log_str = "Best result | "
            for key, val in sorted(eval_results[0].items(), key=lambda x: x[0]): 
                log_str += "{} {} | ".format(key, val)
            tf.logging.info("=" * 20)
            tf.logging.info(log_str)
            res_out.write("Best result:\n")
            res_out.write("{}\t{}\t{}\t{}\n".format(eval_results[0]['global_step'], eval_results[0]['precision'], eval_results[0]['recall'], eval_results[0]['f1']))


if __name__ == "__main__":
    tf.app.run()
