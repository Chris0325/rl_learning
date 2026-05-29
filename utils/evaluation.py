from utils.util import *


def best_action_ratio(action_history, best_actions):
    return np.isin(np.array(action_history), best_actions).cumsum() / np.arange(1, len(action_history)+1)


def experiment(env_func, policy_funcs, runs, horizon, need_best_ratio=False):
    ratio_stats, reward_stats = defaultdict(list), defaultdict(list)
    for policy_func in policy_funcs:
        for _ in tqdm(range(runs), desc=str(policy_func())):
            env, policy = env_func(), policy_func()
            policy.run(env, horizon)
            if need_best_ratio:
                ratio_stats[str(policy)].append(best_action_ratio(policy.value.action_history, env.best_actions()))
            reward_stats[str(policy)].append(policy.value.reward_history)


    plt.subplot(2, 1, 1)
    for policy_func in policy_funcs:
        plt.plot(np.array(reward_stats[str(policy_func())]).mean(axis=0), label=str(policy_func()))
    plt.legend()

    if need_best_ratio:
        plt.subplot(2, 1, 2)
        for policy_func in policy_funcs:
            plt.plot(np.array(ratio_stats[str(policy_func())]).mean(axis=0), label=str(policy_func()))

    plt.legend()
    plt.show()


def banchmark(k, env_func, plans, runs, horizon=1000):
    banchmark_dict = defaultdict(list)
    for plan_name, plan_settings in plans.items():
        for parameter, policy_func in plan_settings:
            reward_stats = []
            for _ in tqdm(range(runs), desc=str(policy_func())):
                env, policy = env_func(), policy_func()
                policy.run(env, horizon)
                reward_stats.append(policy.value.reward_history)
            
            banchmark_dict[plan_name].append([parameter, np.array(reward_stats).mean()])

    for plan_name in plans:
        plt.plot(np.array(banchmark_dict[plan_name])[:, 0], np.array(banchmark_dict[plan_name])[:, 1], label=plan_name)

    plt.legend()
    plt.show()
