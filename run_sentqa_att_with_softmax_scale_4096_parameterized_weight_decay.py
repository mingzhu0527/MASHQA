from run_sentqa_att_with_softmax_scale_4096 import *
# 1. sent_list, 跟"context"在同一个level
# 2. answer_span，跟answer_start在同一个level
# 3. 跟answer_start在同一个level is None for QA with non_consec 

# from run_squad_GPU import *



class AttentionNoPoolingLayer(object):
    
    def __init__(self):
        self.output_layer = None
        
    def __call__(self, k_enc_outputs, q_enc_outputs):
        # k -- embedding dimension
        
        k = k_enc_outputs.shape[2].value
        # dot-product attention
        with tf.variable_scope('attention_layer'):
            # K * Q_t
            proj_q = tf.matmul(k_enc_outputs, q_enc_outputs, transpose_b=True)
            # Normalize (scaled dot product)
            proj_q = tf.nn.softmax(tf.multiply(proj_q, 1/math.sqrt(2048)))
            # (K * Q_t) * Q
            proj_q = tf.matmul(proj_q, q_enc_outputs)
        
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
    
class AttentionPoolingLayer(object):
    
    def __init__(self, init):
        self.output_layer = None
        self.init = init
        
    # 16*b, 32, 1024
    def __call__(self, k_enc_outputs, q_enc_outputs):
        # k -- embedding dimension
#         k = k_enc_outputs.shape[2].value
        # dot-product attention
        with tf.variable_scope('attention_layer'):
            # 13*b, 32, 1024, 
            # 32*1024, 1024*32 -> 32*32
            proj_q0 = tf.matmul(k_enc_outputs, q_enc_outputs, transpose_b=True) # transpose_b means transpose second var
            # Normalize
            # 13*b, 32, 32
#             proj_q1 = tf.nn.softmax(proj_q0)
#             proj_q1 = tf.nn.softmax(tf.multiply(proj_q0, 1/math.sqrt(1024)))
            proj_q1 = tf.nn.softmax(tf.multiply(proj_q0, 1.0/4096.0))
            # 32 * 1024
            proj_q = tf.matmul(proj_q1, q_enc_outputs)
            # 1024 * 32
            proj_q = tf.transpose(proj_q, perm=[0, 2, 1])
        # pooling
        with tf.variable_scope('pooling_layer'):
            proj_q = tf.layers.dense(
                proj_q,
                1,
                kernel_initializer=self.init,
                name="att_dense_1")
#             self.output_layer = Dense(1)
#             proj_q = self.output_layer(proj_q)
            # 16*b, 1024
            proj_q = tf.squeeze(proj_q, axis=2)
        
        return proj_q, tf.reshape(proj_q0, [-1, 1024]), tf.reshape(proj_q1, [-1, 1024])



def _get_spm_basename():
    spm_basename = os.path.basename(FLAGS.spiece_model_file)
    return spm_basename


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
#     hsz = tf.shape(output)[2]
    # output shape: lbh, l = 512, b = batchsize, h = hiddensize = 1024
    # reshape into 16*batchsize*32h, then condense 32h into sentence vector
    
    sent_output = tf.reshape(output, [-1, FLAGS.max_sent_length, bsz, xlnet_config.d_model])
    nsent = tf.shape(sent_output)[0]
    # 16，bsz, 32, hsz
    sent_output = tf.einsum("nsbh->nbsh", sent_output)
    
    # 13, bsz, 32, hsz
    sent_output_slice = tf.slice(sent_output, [0, 0, 0, 0], [FLAGS.max_sent_num, bsz, FLAGS.max_sent_length, xlnet_config.d_model])
    
    # 13*b, 32, 1024
    sent_output = tf.reshape(sent_output_slice, [FLAGS.max_sent_num*bsz, FLAGS.max_sent_length, xlnet_config.d_model])
    
    # nsent*bsz, hsz
    sent_rep_vectors, proj_q0, proj_q1 = AttentionPoolingParameterizedLayer()(sent_output, sent_output)
    # 13, b, 1024
    sent_rep_vectors = tf.reshape(sent_rep_vectors, [FLAGS.max_sent_num, bsz, xlnet_config.d_model])
    sent_rep_out = tf.transpose(sent_rep_vectors, [1, 0, 2])
    proj_q0 = tf.reshape(proj_q0, [FLAGS.max_sent_num, bsz, xlnet_config.d_model])
    proj_q0 = tf.transpose(proj_q0, [1, 0, 2])
    proj_q1 = tf.reshape(proj_q1, [FLAGS.max_sent_num, bsz, xlnet_config.d_model])
    proj_q1 = tf.transpose(proj_q1, [1, 0, 2])
    output_trans = tf.transpose(output, [1, 0, 2])
    sent_output_slice_trans = tf.transpose(sent_output_slice, [1, 0, 2, 3])
#     sent_output = tf.reshape(sent_output, [FLAGS.max_sent_num, bsz, FLAGS.max_sent_length*xlnet_config.d_model])
#     sent_rep_vectors = tf.layers.dense(
#                 sent_output,
#                 1024,
#                 kernel_initializer=initializer,
#                 name="sent_dense_1")
#     sent_rep_vectors = tf.nn.relu(sent_rep_vectors)
#     sent_rep_vectors = tf.nn.dropout(sent_rep_vectors, keep_prob)
    
    cls_index = tf.one_hot(cls_index, seq_len, axis=-1, dtype=tf.float32)
    cls_feature = tf.einsum("lbh,bl->bh", output, cls_index)
    # 1, b, h
    cls_feature = tf.expand_dims(cls_feature, 0)
    cls_tile = tf.tile(cls_feature, [FLAGS.max_sent_num, 1, 1])
    # 13, b, 2h
    sent_rep_cls = tf.concat([sent_rep_vectors, cls_tile], -1)
    sent_rep_cls = tf.reshape(sent_rep_cls, [FLAGS.max_sent_num, bsz, 2*xlnet_config.d_model])
    
    
    # logit of the start position
    with tf.variable_scope("class_logits"):

        start_logits = tf.layers.dense(
                sent_rep_cls,
                512,
                kernel_initializer=initializer,
                name="start_dense_1")
#         start_logits = tf.nn.relu(start_logits)
        start_logits = tf.nn.dropout(start_logits, keep_prob)
        
        start_logits = tf.layers.dense(
                start_logits,
                384,
                kernel_initializer=initializer,
                name="start_dense_2")
#         start_logits = tf.nn.relu(start_logits)
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
    return_dict["sent_rep_vectors"] = sent_rep_out
    return_dict["proj_q0"] = proj_q0
    return_dict["proj_q1"] = proj_q1
    return_dict["output_trans"] = output_trans
    return_dict["sent_output_slice_trans"] = sent_output_slice_trans
    
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
                    "sent_rep_vectors": outputs["sent_rep_vectors"],
                    "proj_q0": outputs["proj_q0"],
                    "proj_q1": outputs["proj_q1"],
                    "output_trans": outputs["output_trans"],
                    "sent_output_slice_trans": outputs["sent_output_slice_trans"],
                
                    
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
        
        
        global_step = tf.train.get_or_create_global_step()
        constant_scale = 20.0
        # decay the scaling factor
        decay_sf = tf.train.polynomial_decay(
                constant_scale,
                global_step=global_step,
                decay_steps=FLAGS.train_steps,
                end_learning_rate=1.0,
                power=2.0)

        scale_factor = tf.where(decay_sf > 1.0, decay_sf, 1.0)
        tf.logging.info("scale_factor shape {}".format(scale_factor.shape))
        
        class_weights = tf.Variable([[1.0, scale_factor]])
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



def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    if FLAGS.do_prepro:
        preprocess()
        preprocess_val()
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

        cur_results = []
        global_acc = 0
        global_case_cnt = 0
        with tf.gfile.Open(os.path.join(FLAGS.predict_dir, "predictions.tsv"), "w") as fout:
#             fout.write("index\tprediction\n")
            for pred_cnt, result in enumerate(estimator.predict(
                    input_fn=eval_input_fn,
                    yield_single_examples=True)):

                if pred_cnt % 1000 == 0:
                    tf.logging.info("Processing example: %d" % (pred_cnt))

                unique_id = int(result["unique_ids"])
                sent_labs = [int(x) for x in result["sent_labels"]]
                num_ans = float(sum(sent_labs))
                pred_ans = []
                for i, (a, lab) in enumerate(zip(result["start_top_log_probs"], sent_labs)):
                    lb0, lb1 = a[0], a[1]
                    label_out = -1
                    if lb1 - lb0 > 0:
                        label_out = 1
                    else:
                        label_out = 0
                    pred_ans.append(label_out)
                    fout.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(pred_cnt, i, label_out, lb0, lb1, lab))

                if num_ans > 0:
                    acc = sum(x[0] * x[1] for x in zip(sent_labs, pred_ans))/num_ans
                    global_case_cnt += 1
                    global_acc += acc
            
            
#             start_top_log_probs = (
#                     [float(x) for x in result["start_top_log_probs"].flat])
#             start_top_index = [int(x) for x in result["start_top_index"].flat]
#             end_top_log_probs = (
#                     [float(x) for x in result["end_top_log_probs"].flat])
#             end_top_index = [int(x) for x in result["end_top_index"].flat]

#             cls_logits = float(result["cls_logits"].flat[0])

#             cur_results.append(
#                     RawResult(
#                             unique_id=unique_id,
#                             start_top_log_probs=start_top_log_probs,
#                             start_top_index=start_top_index,
#                             end_top_log_probs=end_top_log_probs,
#                             end_top_index=end_top_index,
#                             cls_logits=cls_logits))

#         output_prediction_file = os.path.join(
#                 FLAGS.predict_dir, "predictions.json")
#         output_nbest_file = os.path.join(
#                 FLAGS.predict_dir, "nbest_predictions.json")
#         output_null_log_odds_file = os.path.join(
#                 FLAGS.predict_dir, "null_odds.json")

#         ret = write_predictions(eval_examples, eval_features, cur_results,
#                                                         FLAGS.n_best_size, FLAGS.max_answer_length,
#                                                         output_prediction_file,
#                                                         output_nbest_file,
#                                                         output_null_log_odds_file,
#                                                         orig_data)

        # Log current result
        tf.logging.info("=" * 80)
        tf.logging.info("result acc: {}, case count: {}".format(global_acc, global_case_cnt))
        res = -1
        if global_case_cnt != 0:
            res = global_acc/global_case_cnt
        tf.logging.info("result acc {}".format(res))
#         log_str = "Result | "
#         for key, val in ret.items():
#             log_str += "{} {} | ".format(key, val)
#         tf.logging.info(log_str)
        tf.logging.info("=" * 80)


if __name__ == "__main__":
    tf.app.run()
