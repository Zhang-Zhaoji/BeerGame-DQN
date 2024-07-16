from new_env import *

if __name__ == "__main__":
    env = BeerGame() # By default generate random costomer data. 
    AgentTypes = [env.players[i].AgentType for i in range(len(env.players))]
    reward2time = []
    sum_reward2time = []
    print(AgentTypes)
    input("Press enter if the players are you need!")
    # for _episode_index in tqdm(range(int(env.config.maxEpisodesTrain))):
    for _episode_index in tqdm(range(800)):
        obs = env.reset()
        # env.render()
        done = False
        # input()
        cumulative_reward = [0,0,0,0] ####### INIT: prepare to use it in the future!########
        while not done:
            actions = []
            for i in range(4): # get actions for every agent
                env.getAction(k=i)
                #print(env.players[i].action)
                actions.append(int(np.nonzero(env.players[i].action)[0]))
            next_obs, reward, done_list, _ = env.step(actions)
            for j in range(4):
                cumulative_reward[j] += reward[j]

            if "DQN" in AgentTypes: # 对于存在DQN的来说，考虑
                # print("exist DQN!")
                current_states = []
                if env.curTime == 0 or env.curTime == 1:
                    current_states = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
                else: # which means the current state is the former state:
                    current_states = next_state.copy()

                next_state = [[p.IL for p in env.players],[p.OO for p in env.players],[p.AS[env.curTime] for p in env.players],[p.AO[env.curTime] for p in env.players]] 
                # 分别是 current Inventory, current open orders 之前处理的十个任务的东西, arrived shipment, arrived orders
                for k in range(len(AgentTypes)):
                    if AgentTypes[k] == "DQN":
                        if env.config.rewardtype == "total":
                            env.players[k].replaybuffer.add(state=np.array(env.players[k].getCurState(env.curTime-1)),\
                                                            action=actions[k], reward=sum(reward),done=done,\
                                                            next_state = np.array(env.players[k].getCurState(env.curTime))) # 考虑所有人的reward总和
                        elif env.config.rewardtype == "own":
                            env.players[k].replaybuffer.add(state=np.array(env.players[k].getCurState(env.curTime-1)),\
                                                            action=actions[k], reward=reward[k],done=done,\
                                                            next_state = np.array(env.players[k].getCurState(env.curTime))) # 考虑自己的reward
                        if env.players[k].replaybuffer.size() > env.config.minReplayMem:
                            b_s, b_a, b_r, b_ns, b_d = env.players[k].replaybuffer.sample(env.config.batchSize)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'dones': b_d
                            }
                            env.players[k].network.update(transition_dict)
            done = all(done_list)
            # env.render()
        reward2time.append(cumulative_reward)
        sum_reward2time.append(sum(cumulative_reward))
        if _episode_index % 200 == 199:
            for k in range(len(env.players)):
                if AgentTypes[k] == "DQN":
                    env.players[k].network.save_model(_episode_index,agent_index=k)
            plt.plot(sum_reward2time)
            plt.savefig(f"./singleDQN/{_episode_index}_DQNimage.png")
            plt.close()
    with open("DQN_sumreward2time.txt", "w") as file1:
        for num in sum_reward2time:
            file1.write(str(num)+"\n")
    with open("DQN_reward2time.txt", "w") as file2:
        for list0 in reward2time:
            tmp_lst = list(map(str, list0))
            tmp_str = ",".join(tmp_lst)
            file2.write(tmp_str + "\n")