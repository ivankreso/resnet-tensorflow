import numpy as np
import tensorflow as tf
import resnet_model as model
import h5py
import cv2
import time


def evaluate(model, train_dataset, valid_dataset):
  with tf.Graph().as_default():
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    #config.operation_timeout_in_ms = 5000   # terminate on long hangs
    sess = tf.Session(config=config)

    # Build a Graph that computes the logits predictions from the inference model.
    train_ops, init_op, init_feed = model.build(inputs, is_training=True)
    valid_ops = model.build(valid_dataset, is_training=False, reuse=True)
    loss = train_ops[0]

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    if init_op != None:
      sess.run(init_op, feed_dict=init_feed)

    if len(FLAGS.resume_path) > 0:
      print('\nRestoring params from:', FLAGS.resume_path)
      assert tf.gfile.Exists(FLAGS.resume_path)
      resnet_restore = tf.train.Saver(model.variables_to_restore())
      resnet_restore.restore(sess, FLAGS.resume_path)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)

    init_vars = train_helper.get_variables(sess)
    #train_helper.print_variable_diff(sess, init_vars)
    #variable_map = train_helper.get_variable_map()
    num_params = train_helper.get_num_params()
    print('Number of parameters = ', num_params)
    # take the train loss moving average
    #loss_avg_train = variable_map['total_loss/avg:0']
    train_data, valid_data = model.init_eval_data()
    ex_start_time = time.time()
    for epoch_num in range(1, FLAGS.max_epochs + 1):
      print('\ntensorboard --logdir=' + FLAGS.train_dir + '\n')
      train_data['lr'] += [lr.eval(session=sess)]
      num_batches = model.num_examples(train_dataset) // FLAGS.num_validations_per_epoch
      for step in range(num_batches):
      #for step in range(100):
        start_time = time.time()
        run_ops = train_ops + [train_op, global_step]
        #run_ops = [train_op, loss, logits, labels, draw_data, img_name, global_step]
        if step % 300 == 0:
          #run_ops += [summary_op, loss_avg_train]
          run_ops += [summary_op]
          ret_val = sess.run(run_ops)
          loss_val = ret_val[0]
          summary_str = ret_val[-1]
          global_step_val = ret_val[-2]
          summary_writer.add_summary(summary_str, global_step_val)
        else:
          ret_val = sess.run(run_ops)
          loss_val = ret_val[0]
          #train_helper.print_grad_stats(grads_val, grad_tensors)
          #run_metadata = tf.RunMetadata()
          #ret_val = sess.run(run_ops,
          #            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
          #            run_metadata=run_metadata)
          #(_, loss_val, scores, yt, draw_data_val, img_prefix, global_step_val) = ret_val
          #if step > 10:
          #  trace = timeline.Timeline(step_stats=run_metadata.step_stats)
          #  trace_file = open('timeline.ctf.json', 'w')
          #  trace_file.write(trace.generate_chrome_trace_format())
          #  raise 1
        duration = time.time() - start_time

        assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

        ## estimate training accuracy on the last 30% of the epoch
        #if step > int(0.7 * num_batches):
        #  #label_map = scores[0].argmax(2).astype(np.int32)
        #  label_map = scores.argmax(3).astype(np.int32)
        #  #print(label_map.shape)
        #  #print(yt.shape)
        #  cylib.collect_confusion_matrix(label_map.reshape(-1),
        #                                 yt.reshape(-1), conf_mat)
        #img_prefix = img_prefix[0].decode("utf-8")

        #if FLAGS.draw_predictions and step % 50 == 0:
        #  model.draw_prediction('train', epoch_num, step, ret_val)

        if step % 20 == 0:
          examples_per_sec = FLAGS.batch_size / duration
          sec_per_batch = float(duration)

          format_str = '%s: epoch %d, step %d / %d, loss = %.2f \
            (%.1f examples/sec; %.3f sec/batch)'
          #print('lr = ', clr)
          print(format_str % (train_helper.get_expired_time(ex_start_time), epoch_num,
                              step, model.num_examples(train_dataset), loss_val,
                              examples_per_sec, sec_per_batch))
      #train_helper.print_variable_diff(sess, init_vars)
      model.evaluate('valid', sess, epoch_num, valid_ops, valid_dataset, valid_data)
      model.print_results(valid_data)
      #model.plot_results(train_data, valid_data)

    coord.request_stop()
    coord.join(threads)
    sess.close()


  model = helper.import_module('model', FLAGS.model_path)

  if tf.gfile.Exists(FLAGS.train_dir):
    raise ValueError('Train dir exists: ' + FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  stats_dir = os.path.join(FLAGS.train_dir, 'stats')
  tf.gfile.MakeDirs(stats_dir)
  tf.gfile.MakeDirs(FLAGS.debug_dir + '/train/')
  tf.gfile.MakeDirs(FLAGS.debug_dir + '/valid/')
  f = open(os.path.join(stats_dir, 'log.txt'), 'w')
  sys.stdout = train_helper.Logger(sys.stdout, f)

  copyfile(FLAGS.model_path, os.path.join(FLAGS.train_dir, 'model.py'))
  copyfile(FLAGS.config_path, os.path.join(FLAGS.train_dir, 'config.py'))

  print('Experiment dir: ' + FLAGS.train_dir)
  print('Dataset dir: ' + FLAGS.dataset_dir)
  train_dataset = CityscapesDataset(FLAGS.dataset_dir, 'train')
  valid_dataset = CityscapesDataset(FLAGS.dataset_dir, 'val')
  train(model, train_dataset, valid_dataset)



def main(argv=None):
  MODEL_DEPTH = 50
  #MODEL_PATH ='/home/kivan/datasets/pretrained/resnet/ResNet'+str(MODEL_DEPTH)+'.npy'
  data_path = '/home/kivan/datasets/imagenet/ILSVRC2015/numpy/val_data.hdf5'

  img_size = 224
  image = tf.placeholder(tf.float32, [None, img_size, img_size, 3], 'input')
  labels = tf.placeholder(tf.int32, [None], 'label')
  #logits = build(image, labels, is_training=False)
  logits, init_op, init_feed = model.build(image, labels, is_training=False)
  #all_vars = tf.contrib.framework.get_variables()
  #for v in all_vars:
  #  print(v.name)

  sess = tf.Session()
  #sess.run(tf.initialize_all_variables())
  #sess.run(tf.initialize_local_variables())
  sess.run(init_op, feed_dict=init_feed)

  batch_size = 100

  #data = np.load(data_path)
  #data_x = data[0]
  #data_y = data[1]
  h5f = h5py.File(data_path, 'r')
  data_x = h5f['data_x'][()]
  print(data_x.shape)
  data_y = h5f['data_y'][()]
  h5f.close()

  from tensorpack.utils.loadcaffe import get_caffe_pb
  caffepb = get_caffe_pb()
  obj = caffepb.BlobProto()
  mean_file = '/home/kivan/datasets/imagenet/ILSVRC2015/caffe/imagenet_mean.binaryproto'
  with open(mean_file, 'rb') as f:
    obj.ParseFromString(f.read())
  data_mean = np.array(obj.data).reshape((3, 256, 256)).astype('float32')
  data_mean = np.transpose(data_mean, [1,2,0])
  if img_size != data_mean.shape[0]:
    data_mean = cv2.resize(data_mean, (img_size, img_size))
    #data_mean = ski.transform.resize(data_mean, (img_size, img_size),
    #                                 preserve_range=True, order=3)

  data_x = data_x.astype(np.float32)
  #data_x -= data_mean
  #data_mean = np.array([], dtype=np.float32)
  #data_mean = data_x.mean(0)
  #print(data_x.mean((0,1,2)))
  #print(data_x.std((0,1,2)))
  N = data_x.shape[0]
  assert N % batch_size == 0
  num_batches = N // batch_size

  top5_error = tf.nn.in_top_k(logits, labels, 5)
  top5_wrong = 0
  cnt_wrong = 0
  for i in range(num_batches):
    offset = i * batch_size
    batch_x = data_x[offset:offset+batch_size, ...]
    batch_y = data_y[offset:offset+batch_size, ...]
    start_time = time.time()
    logits_val, top5 = sess.run([logits, top5_error], feed_dict={image:batch_x, labels:batch_y})
    duration = time.time() - start_time
    num_examples_per_step = batch_size
    examples_per_sec = num_examples_per_step / duration
    sec_per_batch = float(duration)

    top5_wrong += (top5==0).sum()
    yp = logits_val.argmax(1).astype(np.int32)
    cnt_wrong += (yp != batch_y).sum()
    if i % 10 == 0:
      print('[%d / %d] top1error = %.2f - top5error = %.2f (%.1f examples/sec; %.3f sec/batch)' % (i, num_batches,
            cnt_wrong / ((i+1)*batch_size) * 100, top5_wrong / ((i+1)*batch_size) * 100,
            examples_per_sec, sec_per_batch))
  print(cnt_wrong / N)
  print(top5_wrong / N)

  #eval_on_ILSVRC12(resnet_param, args.eval)


if __name__ == '__main__':
  tf.app.run()

