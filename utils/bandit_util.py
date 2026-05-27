from utils.util import *
from utils.evaluation import *


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


def banchmark(k, bandit_class, plans, runs, bandit_mean=0, need_best_ratio=False, from_step=0):
    banchmark_dict = defaultdict(list)
    for plan_name, plan_settings in plans.items():
        for parameter, (value, policy) in plan_settings:
            ratio_stats, reward_stats = run_plans(k, bandit_class, [(value, policy)], runs, bandit_mean, need_best_ratio)
            banchmark_dict[plan_name].append([parameter, np.array(reward_stats[plan_to_str(value, policy)][from_step:]).mean()])

    for plan_name in plans:
        plt.plot(np.array(banchmark_dict[plan_name])[:, 0], np.array(banchmark_dict[plan_name])[:, 1], label=plan_name)

    plt.legend()
    plt.show()
