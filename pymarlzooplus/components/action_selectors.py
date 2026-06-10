import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
import torch.nn.functional as F


class MultinomialActionSelector:

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(
            args.epsilon_start,
            args.epsilon_finish,
            args.epsilon_anneal_time,
            decay="linear"
        )
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)
        self.action_selector_strategy = args.action_selector_strategy

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        if self.action_selector_strategy == "maser_selector_strategy":
            if not (th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) > 0.99).all():
                return self.select_action(agent_inputs, avail_actions, t_env, test_mode)

        return picked_actions


class EpsilonGreedyActionSelector:

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(
            args.epsilon_start,
            args.epsilon_finish,
            args.epsilon_anneal_time,
            decay="linear"
        )
        self.epsilon = self.schedule.eval(0)
        self.action_selector_strategy = args.action_selector_strategy

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]

        if self.action_selector_strategy == "maser_selector_strategy":
            if not (th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) > 0.99).all():
                return self.select_action(agent_inputs, avail_actions, t_env, test_mode)

        return picked_actions


class SoftPoliciesSelector:

    def __init__(self, args):
        self.args = args
        self.algo_name = args.name

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        m = Categorical(agent_inputs)
        picked_actions = m.sample().long()

        if self.algo_name == "happo":
            log_probs = m.log_prob(picked_actions)
            return picked_actions, log_probs

        return picked_actions


class ICESActionSelector:
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(
            args.epsilon_start,
            args.epsilon_finish,
            args.epsilon_anneal_time,
            decay="linear",
        )
        self.epsilon = self.schedule.eval(0)

    def select_action(
        self,
        agent_inputs,
        int_agent_inputs,
        avail_actions,
        t_env,
        int_ratio,
        test_mode=False,
    ):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = getattr(self.args, "test_noise", 0.0)
            int_ratio = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0] = -float("inf")  # should never be selected!
        masked_int_q_values = int_agent_inputs.clone()
        masked_int_q_values[avail_actions == 0.0] = -float(
            "inf"
        )  # should never be selected!
        masked_int_q_values = F.softmax(masked_int_q_values, dim=-1)

        m = Categorical(masked_int_q_values)
        int_actions = m.sample().long()

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        # behavior_actions
        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_int = (random_numbers < int_ratio).long()
        behavior_actions = (
            pick_int * int_actions + (1 - pick_int) * masked_q_values.max(dim=2)[1]
        )
        picked_actions = (
            pick_random * random_actions + (1 - pick_random) * behavior_actions
        )

        return picked_actions, m.entropy()


REGISTRY = {
    "epsilon_greedy": EpsilonGreedyActionSelector,
    "multinomial": MultinomialActionSelector,
    "soft_policies": SoftPoliciesSelector,
    "epsilon_expl": ICESActionSelector,
}
