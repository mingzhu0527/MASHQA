import sys
from run_squad_GPU import *
from function_builder_GPU import *

flags.DEFINE_integer("max_sent_length", default=32, help="Max sent length")
flags.DEFINE_integer("max_sent_num", default=13, help="Max sent number")

# 1. sent_list, 跟"context"在同一个level
# 2. answer_span，跟answer_start在同一个level
# 3. 跟answer_start在同一个level is None for QA with non_consec 

# from run_squad_GPU import *


class SquadExample(object):
    """A single training/test example for simple sequence classification.

         For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 paragraph_text,
                 sent_starts,
                 orig_answer_text=None,
                 start_position=None,
                 start_poses=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.sent_starts = sent_starts
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.start_poses = start_poses
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (printable_text(self.qas_id))
        s += ", question_text: %s" % (
                printable_text(self.question_text))
        s += ", paragraph_text: [%s]" % (" ".join(self.paragraph_text))
        if self.start_poses:
            s += ", start_poses: %d" % (printable_text(self.start_poses))
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 input_ids,
                 input_mask,
                 p_mask,
                 segment_ids,
                 paragraph_len,
                 cls_index,
                 sent_labels):

        self.unique_id = unique_id
        self.example_index = example_index
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.p_mask = p_mask
        self.segment_ids = segment_ids
        self.paragraph_len = paragraph_len
        self.cls_index = cls_index
        self.sent_labels = sent_labels


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=list(values)))
            return feature

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f
#         print("type(input_ids)", type(feature.input_ids), type(feature.input_ids[0]))
#         print("type(start_poses)", type(feature.start_poses), type(feature.start_poses[0]))
        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_float_feature(feature.input_mask)
        features["p_mask"] = create_float_feature(feature.p_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
#         features["span_sent_start_poses"] = create_int_feature(feature.span_sent_start_poses)
#         features["span_sent_end_poses"] = create_int_feature(feature.span_sent_end_poses)
        features["cls_index"] = create_int_feature([feature.cls_index])

#         if self.is_training:
        features["sent_labels"] = create_int_feature(feature.sent_labels)
#             features["start_poses"] = create_int_feature(feature.start_poses)
#             features["end_poses"] = create_int_feature(feature.end_poses)
#             features["start_positions"] = create_int_feature([feature.start_position])
#             features["end_positions"] = create_int_feature([feature.end_position])
#             impossible = 0
#             if feature.is_impossible:
#                 impossible = 1
#             features["is_impossible"] = create_float_feature([impossible])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()

    

def _get_spm_basename():
    spm_basename = os.path.basename(FLAGS.spiece_model_file)
    return spm_basename


def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            sent_starts = paragraph['sent_starts']
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                orig_answer_text = None
                is_impossible = False
                start_poses=[]
#                 if is_training:
                is_impossible = qa["is_impossible"]
                if (len(qa["answers"]) != 1) and (not is_impossible):
                    raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                if not is_impossible:
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    start_position = answer["answer_start"]
                    start_poses = answer["answer_starts"]
                else:
                    start_position = -1
                    orig_answer_text = ""

                example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        paragraph_text=paragraph_text,
                        sent_starts = sent_starts,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        start_poses = start_poses,
                        is_impossible=is_impossible)
                examples.append(example)

    return examples

# mark sentence separation
# map sentence separation after sentencePiece
# get output
# condense into sentence vectors
# 
def convert_examples_to_features(examples, sp_model, max_seq_length,
                                                                 doc_stride, max_query_length, is_training,
                                                                 output_fn):
    """Loads a data file into a list of `InputBatch`s."""

    cnt_pos, cnt_neg = 0, 0
    unique_id = 1000000000
    max_N, max_M = 1024, 1024
    f = np.zeros((max_N, max_M), dtype=np.float32)

    for (example_index, example) in enumerate(examples):

        if example_index % 100 == 0:
            tf.logging.info('Converting {}/{} pos {} neg {}'.format(
                    example_index, len(examples), cnt_pos, cnt_neg))

        query_tokens = encode_ids(
                sp_model,
                preprocess_text(example.question_text, lower=FLAGS.uncased))

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        paragraph_text = example.paragraph_text
        sents = []
        labels = []
        for start, length in example.sent_starts:
            if is_training:
                is_ans = 0
                for ans_start, ans_length in example.start_poses:
                    if start == ans_start and length == ans_length:
                        is_ans = 1
                        break
                labels.append(is_ans)
            sent = paragraph_text[start:start + length]
            sents.append(sent)
        
        sent_tokens = []
        for sent in sents:
            sent_tok = encode_pieces(
                sp_model,
                preprocess_text(sent, lower=FLAGS.uncased))
            if len(sent_tok) > FLAGS.max_sent_length:
                sent_tok = sent_tok[0:FLAGS.max_sent_length]
            sent_tokens.append(sent_tok)
            
            
        def _piece_to_id(x):
            if six.PY2 and isinstance(x, unicode):
                x = x.encode('utf-8')
            return sp_model.PieceToId(x)

        # list of sentences
        
        sent_tokens_ids = []
        for sent_tok in sent_tokens:
            sent_id_tokens = list(map(_piece_to_id, sent_tok))
            while len(sent_id_tokens) < FLAGS.max_sent_length:
                sent_id_tokens.append(0)
            sent_tokens_ids.append(sent_id_tokens)
            
            
        sent_offset = 0
        sent_count = len(sent_tokens_ids)  
        sent_num_list = []
        while sent_count >= FLAGS.max_sent_num:
            sent_num_list.append(FLAGS.max_sent_num)
            sent_count -= FLAGS.max_sent_num
            
        if sent_count > 0:
            sent_num_list.append(sent_count)
            
        for sent_num in sent_num_list:
            tokens = []
            token_is_max_context = {}
            segment_ids = []
            p_mask = []
            sent_labels = []
            
            for i in range(sent_offset, sent_offset + sent_num):
                tokens += sent_tokens_ids[i]
                segment_ids += [SEG_ID_P] * len(sent_tokens_ids[i])
                p_mask += [0] * len(sent_tokens_ids[i])
            if sent_num < FLAGS.max_sent_num:
                for i in range(FLAGS.max_sent_num - sent_num):
                    tokens += [0] * FLAGS.max_sent_length
                    segment_ids += [SEG_ID_P] * FLAGS.max_sent_length
                    p_mask += [0] * FLAGS.max_sent_length
            assert len(tokens) == FLAGS.max_sent_length * FLAGS.max_sent_num
            if is_training:
                sent_labels = labels[sent_offset:sent_offset + sent_num]
                if sent_num < FLAGS.max_sent_num:
                    sent_labels += [0] * (FLAGS.max_sent_num - sent_num)
                    
            sent_offset += sent_num
            
            paragraph_len = len(tokens)
            
            tokens.append(SEP_ID)
            segment_ids.append(SEG_ID_P)
            p_mask.append(1)

            # note(zhiliny): we put P before Q, context在前，query在后
            # because during pretraining, B is always shorter than A
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(SEG_ID_Q)
                p_mask.append(1)
            tokens.append(SEP_ID)
            segment_ids.append(SEG_ID_Q)
            p_mask.append(1)

            cls_index = len(segment_ids)
            tokens.append(CLS_ID)
            segment_ids.append(SEG_ID_CLS)
            p_mask.append(0)

            input_ids = tokens

            # The mask has 0 for real tokens and 1 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(1)
                segment_ids.append(SEG_ID_PAD)
                p_mask.append(1)

#             print("len(input_ids)", len(input_ids))
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(p_mask) == max_seq_length
       

            if example_index < 20:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info("sent_labels: %s" % " ".join([str(x) for x in sent_labels]))
                tf.logging.info(
                        "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

                
                    # note(zhiliny): With multi processing,
                    # the example_index is actually the index within the current process
                    # therefore we use example_index=None to avoid being used in the future.
                    # The current code does not use example_index of training data.
            if is_training:
                feat_example_index = None
            else:
                feat_example_index = example_index
            
#             print("type(input_ids)", type(input_ids))
#             print("type(start_poses)", type(start_poses))
            feature = InputFeatures(
                    unique_id=unique_id,
                    example_index=feat_example_index,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    p_mask=p_mask,
                    segment_ids=segment_ids,
                    paragraph_len=paragraph_len,
                    cls_index=cls_index,
                    sent_labels=sent_labels)
#                     start_poses=start_poses,
#                     end_poses=end_poses,
#                     span_sent_start_poses=span_sent_start_poses,
#                     span_sent_end_poses=span_sent_end_poses,
#                     start_position=start_position,
#                     end_position=end_position,
#                     is_impossible=span_is_impossible)

            # Run callback
            output_fn(feature)

            unique_id += 1
    tf.logging.info("Total number of instances: {}".format(unique_id))
#             cnt_pos + cnt_neg, cnt_pos, cnt_neg))
#             if span_is_impossible:
#                 cnt_neg += 1
#             else:
# #                 cnt_pos += 1
#     tf.logging.info("Total number of instances: {} = pos {} neg {}".format(
#             cnt_pos + cnt_neg, cnt_pos, cnt_neg))

    
def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #    Doc: the man went to the store and bought a gallon of milk
    #    Span A: the man went to the
    #    Span B: to the store and bought
    #    Span C: and bought a gallon of
    #    ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
    
def _convert_index(index, pos, M=None, is_start=True):
    if index[pos] is not None:
        return index[pos]
    N = len(index)
    rear = pos
    while rear < N - 1 and index[rear] is None:
        rear += 1
    front = pos
    while front > 0 and index[front] is None:
        front -= 1
    assert index[front] is not None or index[rear] is not None
    if index[front] is None:
        if index[rear] >= 1:
            if is_start:
                return 0
            else:
                return index[rear] - 1
        return index[rear]
    if index[rear] is None:
        if M is not None and index[front] < M - 1:
            if is_start:
                return index[front] + 1
            else:
                return M - 1
        return index[front]
    if is_start:
        if index[rear] > index[front] + 1:
            return index[front] + 1
        else:
            return index[rear]
    else:
        if index[rear] > index[front] + 1:
            return index[rear] - 1
        else:
            return index[front]

    
def preprocess():
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(FLAGS.spiece_model_file)
    spm_basename = _get_spm_basename()

    train_rec_file = os.path.join(
            FLAGS.output_dir,
            "{}.{}.slen-{}.qlen-{}.train.tf_record".format(
                    spm_basename, FLAGS.proc_id, FLAGS.max_seq_length,
                    FLAGS.max_query_length))

    tf.logging.info("Read examples from {}".format(FLAGS.train_file))
    train_examples = read_squad_examples(FLAGS.train_file, is_training=True)
    train_examples = train_examples[FLAGS.proc_id::FLAGS.num_proc]

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in the `input_fn`.
    random.shuffle(train_examples)

    tf.logging.info("Write to {}".format(train_rec_file))
    train_writer = FeatureWriter(
            filename=train_rec_file,
            is_training=True)
    convert_examples_to_features(
            examples=train_examples,
            sp_model=sp_model,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=True,
            output_fn=train_writer.process_feature)
    train_writer.close()
    


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
    
    sent_cls_slice = tf.slice(sent_rep_cls, [0, 0, 0], [FLAGS.max_sent_num, bsz, 2048])
    
    
    
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

        cur_results = []
        global_acc = 0
        global_case_cnt = 0
        with tf.gfile.Open(os.path.join(FLAGS.predict_dir, "predictions.tsv"), "w") as fout:
            fout.write("index\tprediction\n")
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
