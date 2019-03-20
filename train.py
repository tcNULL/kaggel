import os  
import numpy as np  
import tensorflow as tf  
import input_data     
import model  
import resnet_v2

  
N_CLASSES = 2  
IMG_W = 224  
IMG_H = 224  
BATCH_SIZE = 32  
CAPACITY = 1000  
MAX_STEP = 100000 
starter_learning_rate = 0.0001
steps_per_decay = 100
decay_factor = 0.99

def losses(logits, labels):  
    with tf.variable_scope('loss') as scope:  
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
                        (logits=logits, labels=labels, name='xentropy_per_example')  
        loss = tf.reduce_mean(cross_entropy, name='loss')  
        tf.summary.scalar(scope.name + '/loss', loss)  
    return loss  
  
def trainning(loss, learning_rate, global_step):  
    with tf.name_scope('optimizer'):  
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)  
        #global_step = tf.Variable(0, name='global_step', trainable=False)  
        train_op = optimizer.minimize(loss, global_step= global_step)  
    return train_op  
  
def evaluation(logits, labels):  
    with tf.variable_scope('accuracy') as scope:  
        correct = tf.nn.in_top_k(logits, labels, 1)  
        correct = tf.cast(correct, tf.float16)  
        accuracy = tf.reduce_mean(correct)  
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

def run_training():  
    
    train_dir = 'G:/kaggel-cancer-detection/train_jpg/'    
    logs_train_dir = 'G:/kaggel-cancer-detection-code/SaveNet-ResNet/'  

    # 获取图片和标签集
    train, train_label = input_data.get_files(train_dir)  
    # 生成批次
    train_batch, train_label_batch = input_data.get_batch(train,  
                                                          train_label,  
                                                          IMG_W,  
                                                          IMG_H,  
                                                          BATCH_SIZE,   
                                                          CAPACITY)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate = starter_learning_rate,
                                           global_step = global_step,
                                           decay_steps = steps_per_decay,
                                           decay_rate = decay_factor,
                                           staircase = True,#If `True` decay the learning rate at discrete intervals
                                           #staircase = False,change learning rate at every step
                                           )
    print("Enter Model！")
    train_logits, end_points= resnet_v2.resnet_v2_50(inputs = train_batch, num_classes= N_CLASSES) 
    print("Get loss!")
    train_loss = losses(train_logits, train_label_batch)
    print("Start training!")
    train_op = trainning(train_loss, learning_rate, global_step)
    print("Get Acc!")
    train__acc = evaluation(train_logits, train_label_batch)  
    summary_op = tf.summary.merge_all()  
    #sess = tf.Session()
    print ("Save summary")
    
    saver = tf.train.Saver()  
      
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph) 
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                print("break!!!")
                break  
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])#训练
            if sess.run(global_step) % 50 == 0:  
                print('Step %d, train loss = %.2f, learning rate = %.10f,train accuracy = %.2f%%' 
                    %(sess.run(global_step),  tra_loss, sess.run(learning_rate), tra_acc*100.0))  
                #print('global_step:',sess.run(global_step))
                #print('learning rate:',sess.run(learning_rate))
                summary_str = sess.run(summary_op)  
                train_writer.add_summary(summary_str, step) 
            if sess.run(global_step) % 2000 == 0 or (sess.run(global_step)) == MAX_STEP:  
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')  
                saver.save(sess, checkpoint_path, global_step=sess.run(global_step))  
        coord.join(threads)  
    sess.close() 
# train
run_training()