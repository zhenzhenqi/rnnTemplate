# train.py

from __future__ import print_function
from json_checkpoint_vars import dump_checkpoints
from model import Model 
from utils import TextLoader
import pickle
import time
import argparse
import logging
import tensorflow as tf
import os
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default='data/input.txt', help='file path to .txt file')
    parser.add_argument('--save_model', type=str, default='models', help='directory to store the ml5js model')
    parser.add_argument('--save_checkpoints', type=str, default='checkpoints', help='directory to store checkpointed models')
    parser.add_argument('--log_dir', type=str, default='logs', help='directory to store tensorboard logs')
    parser.add_argument('--rnn_size', type=int, default=128, help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm', help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50, help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50, help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--print_every', type=int, default=1, help='print frequency')
    parser.add_argument('--save_every', type=int, default=1000, help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97, help='decay rate for rmsprop')
    parser.add_argument('--output_keep_prob', type=float, default=1.0, help='probability of keeping weights in the hidden layer')
    parser.add_argument('--input_keep_prob', type=float, default=1.0, help='probability of keeping weights in the input layer')
    args = parser.parse_args()
    train(args)

def train(args):
    model_name = os.path.splitext(os.path.basename(args.data_path))[0]
    
    args.save_dir = os.path.join(args.save_checkpoints, model_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    data_loader = TextLoader(args.data_path, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size
    
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        pickle.dump((data_loader.chars, data_loader.vocab), f)

    model = Model(args)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, args.save_dir, max_to_keep=5)
    
    if manager.latest_checkpoint:
        print(f"Restoring from {manager.latest_checkpoint}")
        ckpt.restore(manager.latest_checkpoint).expect_partial()

    summary_writer = tf.summary.create_file_writer(
        os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))

    @tf.function
    def train_step(inputs, targets, initial_state):
        with tf.GradientTape() as tape:
            logits, final_state = model(inputs, states=initial_state, training=True)
            loss = loss_fn(tf.reshape(targets, [-1]), tf.reshape(logits, [-1, args.vocab_size]))
        
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, final_state

    print("Starting training...")
    global_step = optimizer.iterations.numpy()
    for e in range(args.num_epochs):
        optimizer.learning_rate.assign(args.learning_rate * (args.decay_rate ** e))
        data_loader.reset_batch_pointer()
        state = model.get_initial_state(args.batch_size)

        for b in range(data_loader.num_batches):
            start = time.time()
            x, y = data_loader.next_batch()
            train_loss, state = train_step(x, y, state)
            end = time.time()
            
            with summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss, step=global_step)
                tf.summary.scalar('learning_rate', optimizer.learning_rate, step=global_step)

            if global_step % args.print_every == 0:
                print(f"{global_step}/{args.num_epochs * data_loader.num_batches} (epoch {e}), "
                      f"train_loss = {train_loss.numpy():.3f}, time/batch = {end - start:.3f}")

            if global_step % args.save_every == 0 or (e == args.num_epochs-1 and b == data_loader.num_batches-1):
                save_path = manager.save(checkpoint_number=global_step)
                print(f"Model saved to {save_path}!")

            global_step += 1
    
    print("Dumping final model for ml5.js...")
    dump_checkpoints(model, args.save_checkpoints, args.save_model, model_name)

if __name__ == '__main__':
    main()