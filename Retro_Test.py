import retro
import tensorflow as tf
import numpy as np
from skimage import transform
from skimage.color import rgb2gray
from collections import deque
import random
from Memory import Memory
from DQnet import DQnet
env = retro.make(game='SpaceInvaders-Atari2600')

# print("The size of our frame is: ", env.observation_space)
# print("The action size is : ", env.action_space.n)


# state = env.reset()
# print(state)



def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


def preprocess_frame(frame):
    gray = rgb2gray(frame)
    cropped_frame = gray[8:-12, 4:-12]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(cropped_frame, [110, 84])
    # print("transformed")
    return preprocessed_frame
#110*84 plus 4 frame stacking
state_size = [110, 84, 4]
action_size = env.action_space.n
learning_rate =  0.00025  #alpha
total_episodes = 50
max_steps = 50000  #max action numbers
batch_size = 64
#???
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00001
# discount factor
gamma = 0.9
pretrain_length = batch_size
memory_size = 1000000
training = True
episode_render = False
stack_size = 4  # We stack 4 frames
# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
memory = Memory(max_size = memory_size)
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
# print(possible_actions)
DQNetwork = DQnet(state_size, action_size,learning_rate,name="DQNet")

def intiMemoryStack(env,batch_size,stacked_frames,memory_buffer,possible_action,action_size):
    for i in range(batch_size):
        if i ==0:
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames,state,True)
        Random_choice = random.randint(1, action_size)-1
        Random_Action = possible_action[Random_choice]
        next_state, rewards, done,_= env.step(Random_Action)
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        if not done:
            memory_buffer.add((state,Random_Action, rewards,next_state,done))
            state= next_state
        else:
            next_state = np.zeros(state.shape)
            memory_buffer.add((state, Random_Action, rewards, next_state, done))
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True)


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        choice = random.randint(1, len(possible_actions)) - 1
        action = possible_actions[choice]

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[choice]

    return action, explore_probability


saver = tf.train.Saver()
intiMemoryStack(env,batch_size,stacked_frames,memory,possible_actions,action_size)
if training == True:
    # config = tf.ConfigProto()

    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
     #try various numbers here
    # with tf.Session() as sess:
    with tf.Session(config=session_config) as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        for episode in range(total_episodes):
            # Set step to 0
            step = 0

            # Initialize the rewards of the episode
            episode_rewards = []

            # Make a new episode and observe the first state
            state = env.reset()

            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < max_steps:
                step += 1

                # Increase decay_step
                decay_step += 1

                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                             possible_actions)

                # Perform the action and get the next_state, reward, and done information
                next_state, reward, done, _ = env.step(action)

                if episode_render:
                    env.render()

                episode_rewards.append(reward)

                if done:

                    next_state = np.zeros((110, 84), dtype=np.int)

                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    step = max_steps

                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Explore P: {:.4f}'.format(explore_probability),
                          'Training Loss {:.4f}'.format(loss))

                    memory.add((state, action, reward, next_state, done))

                else:

                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    memory.add((state, action, reward, next_state, done))
                    state = next_state
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])
                target_Qs_batch = []


                Qs_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})

                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                   feed_dict={DQNetwork.inputs_: states_mb,
                                              DQNetwork.target_Q: targets_mb,
                                              DQNetwork.actions_: actions_mb})

            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")
